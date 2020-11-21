//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/backend/VPUIP.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/VPUIP/schema.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/StandardTypes.h>

#include <precision_utils.h>

#include <unordered_map>

using namespace vpux;

namespace {

flatbuffers::Offset<MVCNN::Version> createVersion(VPUIP::BlobWriter& writer) {
    MVCNN::VersionBuilder builder(writer);
    builder.add_majorV(3);
    builder.add_minorV(11);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Resources>
        serializeResources(VPUIP::BlobWriter& writer,
                           VPUIP::ResourcesAttr resources) {
    MVCNN::ResourcesBuilder builder(writer);
    builder.add_upa_shaves(
            checked_cast<uint32_t>(resources.upa_shaves().getInt()));
    builder.add_nce2_blocks(
            checked_cast<uint32_t>(resources.nce2_blocks().getInt()));
    builder.add_upa_shared_cmx(
            checked_cast<uint32_t>(resources.upa_shared_cmx().getInt()));
    builder.add_nn_cmx_per_slice(
            checked_cast<uint32_t>(resources.nn_cmx_per_slice().getInt()));
    builder.add_nn_cmx_slice_amount(
            checked_cast<uint32_t>(resources.nn_cmx_slice_amount().getInt()));
    builder.add_ddr_scratch(
            checked_cast<uint32_t>(resources.ddr_scratch().getInt()));
    builder.add_csram_storage(
            checked_cast<uint32_t>(resources.csram_storage().getInt()));
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::SummaryHeader>
        createSummaryHeader(VPUIP::BlobWriter& writer,
                            VPUIP::GraphOp graphOp,
                            mlir::FuncOp graphFunc,
                            ptrdiff_t taskCount) {
    auto inputsInfo =
            to_vector<1>(graphOp.inputsInfo().getOps<VPUIP::TensorInfoOp>());
    auto outputsInfo =
            to_vector<1>(graphOp.outputsInfo().getOps<VPUIP::TensorInfoOp>());

    SmallVector<VPUIP::BlobWriter::TensorReference, 1> graphInputs, userInputs;
    graphInputs.reserve(inputsInfo.size());
    userInputs.reserve(inputsInfo.size());

    for (const auto& p : inputsInfo | indexed) {
        auto userInfo = p.value();
        auto val = graphFunc.getArgument(checked_cast<uint32_t>(p.index()));

        const auto graphType = val.getType().cast<mlir::MemRefType>();
        const auto userType = mlir::MemRefType::get(graphType.getShape(),
                                                    userInfo.precision(),
                                                    {userInfo.layout()});

        graphInputs.push_back(
                writer.createTensor(val,
                                    userInfo.name(),
                                    VPUIP::MemoryLocation::ProgrammableInput,
                                    0));
        userInputs.push_back(
                writer.createTensor(userInfo.name(),
                                    userType,
                                    VPUIP::MemoryLocation::ProgrammableInput,
                                    0));
    }

    SmallVector<VPUIP::BlobWriter::TensorReference, 1> graphOutputs,
            userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto funcArgInd = inputsInfo.size() + p.index();

        auto userInfo = p.value();
        auto val = graphFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto graphType = val.getType().cast<mlir::MemRefType>();
        const auto userType = mlir::MemRefType::get(graphType.getShape(),
                                                    userInfo.precision(),
                                                    {userInfo.layout()});

        graphOutputs.push_back(
                writer.createTensor(val,
                                    userInfo.name(),
                                    VPUIP::MemoryLocation::ProgrammableOutput,
                                    0));
        userOutputs.push_back(
                writer.createTensor(userInfo.name(),
                                    userType,
                                    VPUIP::MemoryLocation::ProgrammableOutput,
                                    0));
    }

    SmallVector<int8_t, 1> options;
    if (VPUIP::bitEnumContains(graphOp.options(),
                               VPUIP::ExecutionFlag::DynamicBarriers)) {
        options.push_back(
                static_cast<int8_t>(MVCNN::ExecutionFlag_DynamicBarriers));
    }
    const auto serializedOptions = writer.createVector(options);

    const auto serializedVersion = createVersion(writer);
    const auto serializedName = writer.createString(graphOp.identifier());
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources =
            serializeResources(writer, graphOp.resources());

    MVCNN::SummaryHeaderBuilder builder(writer);
    builder.add_version(serializedVersion);
    builder.add_identifier(serializedName);
    builder.add_net_input(serializedGraphInputs);
    builder.add_net_output(serializedGraphOutputs);
    builder.add_task_count(checked_cast<uint32_t>(taskCount));
    builder.add_options(serializedOptions);
    builder.add_resources(serializedResources);
    builder.add_in_tensor_desc(serializedUserInputs);
    builder.add_out_tensor_desc(serializedUserOutputs);
    return builder.Finish();
}

}  // namespace

flatbuffers::DetachedBuffer vpux::VPUIP::exportToBlob(mlir::ModuleOp module) {
    VPUIP::GraphOp graphOp;
    mlir::FuncOp graphFunc;
    VPUX_THROW_UNLESS(
            mlir::succeeded(
                    VPUIP::GraphOp::getFromModule(module, graphOp, graphFunc)),
            "Invalid VPUIP Dialect IR");

    BlobWriter writer;

    const auto allTasks = graphFunc.getOps<VPUIP::TaskOpInterface>();
    const auto taskCount = std::distance(allTasks.begin(), allTasks.end());

    const auto header =
            createSummaryHeader(writer, graphOp, graphFunc, taskCount);

    using TaskList = std::vector<BlobWriter::Task>;
    using TaskListMap = EnumMap<VPUIP::TaskType, TaskList>;
    TaskListMap tasksMap;

    std::vector<BlobWriter::BinaryData> binaryData;

    const auto callback = [&](mlir::Operation* op) {
        if (auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op)) {
            tasksMap[task.getTaskType()].push_back(writer.createTask(task));
        } else if (auto tensorOp = mlir::dyn_cast<DeclareTensorOp>(op)) {
            VPUX_THROW_UNLESS(tensorOp.offset().hasValue(),
                              "Memory for Operation {0} was not allocated",
                              *op);

            writer.createTensor(tensorOp.memory(),
                                "",
                                tensorOp.location(),
                                tensorOp.offset().getValue());
        } else if (auto barrierOp = mlir::dyn_cast<DeclareBarrierOp>(op)) {
            writer.createBarrier(barrierOp.barrier());
        } else if (mlir::dyn_cast<mlir::ReturnOp>(op) != nullptr ||
                   op == graphFunc.getOperation()) {
            // do nothing
        } else {
            VPUX_THROW("Unknown Operation {0}", *op);
        }
    };

    graphFunc.walk(callback);

    std::vector<BlobWriter::TaskList> taskLists;
    taskLists.reserve(tasksMap.size());
    for (const auto& taskList : tasksMap | map_values) {
        const auto serializedTaskList = writer.createVector(taskList);

        MVCNN::TaskListBuilder builder(writer);
        builder.add_content(serializedTaskList);
        taskLists.push_back(builder.Finish());
    }

    const auto serializedTaskLists = writer.createVector(taskLists);

    const auto barrierTable = writer.createVector(writer.getAllBarriers());

    const auto serializedBinaryData = writer.createVector(binaryData);

    MVCNN::GraphFileBuilder graphBuilder(writer);
    graphBuilder.add_header(header);
    graphBuilder.add_task_lists(serializedTaskLists);
    graphBuilder.add_barrier_table(barrierTable);
    graphBuilder.add_binary_data(serializedBinaryData);
    const auto serializedGraph = graphBuilder.Finish();

    writer.impl().Finish(serializedGraph, "BLOB");

    return writer.impl().Release();
}
