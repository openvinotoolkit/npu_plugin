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
#include "vpux/compiler/dialect/VPUIP/schema.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

#include <precision_utils.h>

#include <unordered_map>

using namespace vpux;

namespace {

flatbuffers::Offset<MVCNN::Version> createVersion(VPUIP::BlobWriter& writer, VPUIP::VersionAttr version) {
    const auto serializedHash = writer.createString(version.hash().getValue());
    const auto serializedContext = writer.createString(version.contextStr().getValue());

    MVCNN::VersionBuilder builder(writer);
    builder.add_majorV(version.majorV().getInt());
    builder.add_minorV(version.minorV().getInt());
    builder.add_patchV(version.patchV().getInt());
    builder.add_hash(serializedHash);
    builder.add_context(serializedContext);
    return builder.Finish();
}

MVCNN::PhysicalProcessor createPhysicalProcessor(VPUIP::PhysicalProcessor proc) {
    switch (proc) {
    case VPUIP::PhysicalProcessor::ARM:
        return MVCNN::PhysicalProcessor_ARM;
    case VPUIP::PhysicalProcessor::Leon_RT:
        return MVCNN::PhysicalProcessor_LEON_RT;
    case VPUIP::PhysicalProcessor::Leon_NN:
        return MVCNN::PhysicalProcessor_LEON_NN;
    case VPUIP::PhysicalProcessor::SHAVE_UPA:
        return MVCNN::PhysicalProcessor_UPA_SHV;
    case VPUIP::PhysicalProcessor::SHAVE_NN:
        return MVCNN::PhysicalProcessor_NN_SHV;
    case VPUIP::PhysicalProcessor::NCE_Cluster:
        return MVCNN::PhysicalProcessor_NCE_Cluster;
    case VPUIP::PhysicalProcessor::NCE_PerClusterDPU:
        return MVCNN::PhysicalProcessor_NCE_PerClusterDPU;
    default:
        VPUX_THROW("Unsupported PhysicalProcessor '{0}'", proc);
    }
}

flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorMapping(VPUIP::BlobWriter& writer,
                                                                    VPUIP::ProcessorMappingAttr attr) {
    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(attr.item().getValue()));
    builder.add_number(attr.number().getInt());
    builder.add_is_bitmask(attr.isBitMask().getValue());
    return builder.Finish();
}

MVCNN::PhysicalMem createPhysicalMem(VPUIP::PhysicalMemory mem) {
    switch (mem) {
    case VPUIP::PhysicalMemory::DDR:
        return MVCNN::PhysicalMem_DDR;
    case VPUIP::PhysicalMemory::CSRAM:
        return MVCNN::PhysicalMem_CSRAM;
    case VPUIP::PhysicalMemory::CMX_UPA:
        return MVCNN::PhysicalMem_UPA_CMX;
    case VPUIP::PhysicalMemory::CMX_NN:
        return MVCNN::PhysicalMem_NN_CMX;
    default:
        VPUX_THROW("Unsupported PhysicalMemory '{0}'", mem);
    }
}

flatbuffers::Offset<MVCNN::MemoryMapping> createMemoryMapping(VPUIP::BlobWriter& writer,
                                                              VPUIP::MemoryMappingAttr attr) {
    MVCNN::MemoryMappingBuilder builder(writer);
    builder.add_item(createPhysicalMem(attr.item().getValue()));
    builder.add_number(attr.number().getInt());
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::MemoryRelationshipMapping> createMemoryRelationshipMapping(
        VPUIP::BlobWriter& writer, VPUIP::MemoryRelationshipMappingAttr attr) {
    MVCNN::MemoryRelationshipMappingBuilder builder(writer);
    builder.add_from_item(createPhysicalMem(attr.fromItem().getValue()));
    builder.add_to_item(createPhysicalMem(attr.toItem().getValue()));
    builder.add_number(attr.number().getValueAsDouble());
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Resources> createResources(VPUIP::BlobWriter& writer, VPUIP::ResourcesAttr resources) {
    const auto processor_allocation =
            writer.createVector(resources.processor_allocation().getAsRange<VPUIP::ProcessorMappingAttr>() |
                                transformed([&](VPUIP::ProcessorMappingAttr attr) {
                                    return createProcessorMapping(writer, attr);
                                }));

    const auto processor_frequencies =
            writer.createVector(resources.processor_frequencies().getAsRange<VPUIP::ProcessorMappingAttr>() |
                                transformed([&](VPUIP::ProcessorMappingAttr attr) {
                                    return createProcessorMapping(writer, attr);
                                }));

    const auto memory_sizes = writer.createVector(resources.memory_sizes().getAsRange<VPUIP::MemoryMappingAttr>() |
                                                  transformed([&](VPUIP::MemoryMappingAttr attr) {
                                                      return createMemoryMapping(writer, attr);
                                                  }));

    const auto memory_bandwidth =
            writer.createVector(resources.memory_bandwidth().getAsRange<VPUIP::MemoryRelationshipMappingAttr>() |
                                transformed([&](VPUIP::MemoryRelationshipMappingAttr attr) {
                                    return createMemoryRelationshipMapping(writer, attr);
                                }));

    MVCNN::ResourcesBuilder builder(writer);
    builder.add_processor_allocation(processor_allocation);
    builder.add_processor_frequencies(processor_frequencies);
    builder.add_memory_sizes(memory_sizes);
    builder.add_memory_bandwidth(memory_bandwidth);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(VPUIP::BlobWriter& writer, VPUIP::GraphOp graphOp,
                                                              mlir::FuncOp graphFunc, ptrdiff_t taskCount) {
    auto inputsInfo = to_vector<1>(graphOp.inputsInfo().getOps<VPUIP::TensorInfoOp>());
    auto outputsInfo = to_vector<1>(graphOp.outputsInfo().getOps<VPUIP::TensorInfoOp>());

    SmallVector<VPUIP::BlobWriter::TensorReference, 1> graphInputs, userInputs;
    graphInputs.reserve(inputsInfo.size());
    userInputs.reserve(inputsInfo.size());

    for (const auto& p : inputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        auto userInfo = p.value();
        auto val = graphFunc.getArgument(ind);

        const auto graphType = val.getType().cast<mlir::MemRefType>();
        const auto userType = mlir::MemRefType::get(graphType.getShape(), userInfo.precision(), {userInfo.layout()});

        graphInputs.push_back(
                writer.createTensor(val, userInfo.name(), VPUIP::MemoryLocation::ProgrammableInput, ind, 0));

        userInputs.push_back(
                writer.createTensor(userInfo.name(), userType, VPUIP::MemoryLocation::ProgrammableInput, ind, 0));
    }

    SmallVector<VPUIP::BlobWriter::TensorReference, 1> graphOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        const auto funcArgInd = inputsInfo.size() + p.index();

        auto userInfo = p.value();
        auto val = graphFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto graphType = val.getType().cast<mlir::MemRefType>();
        const auto userType = mlir::MemRefType::get(graphType.getShape(), userInfo.precision(), {userInfo.layout()});

        graphOutputs.push_back(
                writer.createTensor(val, userInfo.name(), VPUIP::MemoryLocation::ProgrammableOutput, ind, 0));

        userOutputs.push_back(
                writer.createTensor(userInfo.name(), userType, VPUIP::MemoryLocation::ProgrammableOutput, ind, 0));
    }

    SmallVector<int8_t, 1> options;
    if (VPUIP::bitEnumContains(graphOp.options(), VPUIP::ExecutionFlag::DynamicBarriers)) {
        options.push_back(static_cast<int8_t>(MVCNN::ExecutionFlag_DynamicBarriers));
    }
    const auto serializedOptions = writer.createVector(options);

    const auto serializedVersion = createVersion(writer, graphOp.version());
    const auto serializedName = writer.createString(graphOp.identifier());
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources = createResources(writer, graphOp.resources());

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

flatbuffers::DetachedBuffer vpux::VPUIP::exportToBlob(mlir::ModuleOp module, Logger log) {
    log.setName("VPUIP::BackEnd");

    log.trace("Extract {0} from Module", VPUIP::GraphOp::getOperationName());
    VPUIP::GraphOp graphOp;
    mlir::FuncOp graphFunc;
    VPUX_THROW_UNLESS(mlir::succeeded(VPUIP::GraphOp::getFromModule(module, graphOp, graphFunc)),
                      "Invalid VPUIP Dialect IR");

    BlobWriter writer(log.nest());

    const auto allTasks = graphFunc.getOps<VPUIP::TaskOpInterface>();
    const auto taskCount = std::distance(allTasks.begin(), allTasks.end());

    const auto header = createSummaryHeader(writer, graphOp, graphFunc, taskCount);

    using TaskList = std::vector<BlobWriter::Task>;
    using TaskListMap = EnumMap<VPUIP::TaskType, TaskList>;
    TaskListMap tasksMap;

    std::vector<BlobWriter::BinaryData> binaryData;

    size_t tempTensorInd = 0;
    size_t constantTensorInd = 0;
    const auto callback = [&](mlir::Operation* op) {
        if (auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op)) {
            tasksMap[task.getTaskType()].push_back(writer.createTask(task));
        } else if (auto tensorOp = mlir::dyn_cast<DeclareTensorOp>(op)) {
            writer.createTensor(tensorOp.memory(), llvm::formatv("temp-{0}", tempTensorInd).str(), tensorOp.locale(),
                                tensorOp.localeIndex(), tensorOp.dataIndex(), tensorOp.sparsityIndex(),
                                tensorOp.storageElementIndex(), tensorOp.storageElementSize(), tensorOp.leadingOffset(),
                                tensorOp.trailingOffset());

            ++tempTensorInd;
        } else if (auto tensorOp = mlir::dyn_cast<DeclareConstantTensorOp>(op)) {
            writer.createBinaryData(tensorOp.content(), tensorOp.csramCacheable());

            writer.createTensor(tensorOp.memory(), llvm::formatv("constant-{0}", tempTensorInd).str(),
                                MemoryLocation::GraphFile, constantTensorInd, 0);

            ++constantTensorInd;
        } else if (auto barrierOp = mlir::dyn_cast<DeclareBarrierOp>(op)) {
            writer.createBarrier(barrierOp.barrier());
        } else if (mlir::dyn_cast<mlir::ReturnOp>(op) != nullptr || op == graphFunc.getOperation()) {
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
