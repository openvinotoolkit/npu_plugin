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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
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
    builder.add_majorV(checked_cast<uint32_t>(version.majorV().getInt()));
    builder.add_minorV(checked_cast<uint32_t>(version.minorV().getInt()));
    builder.add_patchV(checked_cast<uint32_t>(version.patchV().getInt()));
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
                                                                    IERT::ExecutorResourceOp res) {
    const auto kind = res.kindAttr().dyn_cast_or_null<VPUIP::PhysicalProcessorAttr>();
    VPUX_THROW_UNLESS(kind != nullptr, "Got unknown executor kind '{0}'", res.kindAttr());

    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(kind.getValue()));
    builder.add_number(checked_cast<double>(res.count()));
    builder.add_is_bitmask(false);
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

flatbuffers::Offset<MVCNN::MemoryMapping> createMemoryMapping(VPUIP::BlobWriter& writer, IERT::MemoryResourceOp res) {
    const auto kind = res.kindAttr().dyn_cast_or_null<VPUIP::PhysicalMemoryAttr>();
    VPUX_THROW_UNLESS(kind != nullptr, "Got unknown memory space kind '{0}'", res.kindAttr());

    MVCNN::MemoryMappingBuilder builder(writer);
    builder.add_item(createPhysicalMem(kind.getValue()));
    builder.add_number(checked_cast<double>(res.byteSize()));
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Resources> createResources(VPUIP::BlobWriter& writer, mlir::ModuleOp module) {
    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    VPUX_THROW_UNLESS(resources != nullptr, "Missing IERT run-time resources information");

    const auto usedMemory = writer.createVector(resources.usedMemory().getOps<IERT::MemoryResourceOp>() |
                                                transformed([&](IERT::MemoryResourceOp res) {
                                                    return createMemoryMapping(writer, res);
                                                }));

    const auto usedExecutors = writer.createVector(resources.usedExecutors().getOps<IERT::ExecutorResourceOp>() |
                                                   transformed([&](IERT::ExecutorResourceOp res) {
                                                       return createProcessorMapping(writer, res);
                                                   }));

    MVCNN::ResourcesBuilder builder(writer);
    builder.add_processor_allocation(usedExecutors);
    builder.add_memory_sizes(usedMemory);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(VPUIP::BlobWriter& writer, mlir::ModuleOp module,
                                                              VPUIP::GraphOp graphOp, IE::CNNNetworkOp netOp,
                                                              mlir::FuncOp netFunc, ptrdiff_t taskCount) {
    auto inputsInfo = netOp.getInputsInfo();
    auto outputsInfo = netOp.getOutputsInfo();

    SmallVector<VPUIP::BlobWriter::TensorReference> graphInputs, userInputs;
    graphInputs.reserve(inputsInfo.size());
    userInputs.reserve(inputsInfo.size());

    for (const auto& p : inputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(ind);

        const auto userType = userInfo.userType().cast<mlir::MemRefType>();

        graphInputs.push_back(
                writer.createTensor(val, userInfo.name(), VPUIP::MemoryLocation::ProgrammableInput, ind, 0));

        userInputs.push_back(
                writer.createTensor(userInfo.name(), userType, VPUIP::MemoryLocation::ProgrammableInput, ind, 0));
    }

    SmallVector<VPUIP::BlobWriter::TensorReference> graphOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        const auto funcArgInd = inputsInfo.size() + p.index();

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto userType = userInfo.userType().cast<mlir::MemRefType>();

        graphOutputs.push_back(
                writer.createTensor(val, userInfo.name(), VPUIP::MemoryLocation::ProgrammableOutput, ind, 0));

        userOutputs.push_back(
                writer.createTensor(userInfo.name(), userType, VPUIP::MemoryLocation::ProgrammableOutput, ind, 0));
    }

    SmallVector<int8_t> options;
    if (VPUIP::bitEnumContains(graphOp.options(), VPUIP::ExecutionFlag::DynamicBarriers)) {
        options.push_back(static_cast<int8_t>(MVCNN::ExecutionFlag_DynamicBarriers));
    }
    const auto serializedOptions = writer.createVector(options);

    const auto serializedVersion = createVersion(writer, graphOp.version());
    const auto serializedName = writer.createString(module.getName().getValueOr("network"));
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources = createResources(writer, module);

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

    log.trace("Extract 'IE.{0}' from Module", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Extract 'VPUIP.{0}' from Module", VPUIP::GraphOp::getOperationName());
    auto graphOp = VPUIP::GraphOp::getFromModule(module);

    BlobWriter writer(log.nest());

    const auto allTasks = netFunc.getOps<VPUIP::TaskOpInterface>();
    const auto taskCount = std::distance(allTasks.begin(), allTasks.end());

    const auto header = createSummaryHeader(writer, module, graphOp, netOp, netFunc, taskCount);

    using TaskList = std::vector<BlobWriter::Task>;
    using TaskListMap = EnumMap<VPUIP::TaskType, TaskList>;
    TaskListMap tasksMap;

    std::vector<BlobWriter::BinaryData> binaryData;

    size_t tempTensorInd = 0;
    size_t constantTensorInd = 0;
    const auto callback = [&](mlir::Operation* op) {
        log.trace("Serialize Operation '{0}' at '{1}'", op->getName(), op->getLoc());

        if (auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(op)) {
            log.nest().trace("Got '{0}' Task", task.getTaskType());

            tasksMap[task.getTaskType()].push_back(writer.createTask(task));
        } else if (auto tensorOp = mlir::dyn_cast<DeclareTensorOp>(op)) {
            writer.createTensor(tensorOp.memory(), llvm::formatv("temp-{0}", tempTensorInd).str(), tensorOp.locale(),
                                tensorOp.localeIndex(), tensorOp.dataIndex(), tensorOp.sparsityIndex(),
                                tensorOp.storageElementIndex(), tensorOp.storageElementSize(), tensorOp.leadingOffset(),
                                tensorOp.trailingOffset());

            ++tempTensorInd;
        } else if (auto constOp = mlir::dyn_cast<DeclareConstantTensorOp>(op)) {
            log.nest().trace("Got constant with actual type '{0} and storage type '{1}'", constOp.getActualType(),
                             constOp.getContentType());

            const auto content = constOp.getContent();
            const auto actualType = constOp.getType();
            const auto csramCacheable = constOp.csramCacheable();

            const auto binData = writer.createBinaryData(content, actualType, csramCacheable);
            binaryData.push_back(binData);

            writer.createTensor(constOp.output(), llvm::formatv("constant-{0}", constantTensorInd).str(),
                                MemoryLocation::GraphFile, checked_cast<uint32_t>(constantTensorInd), 0);

            ++constantTensorInd;
        } else if (auto barrierOp = mlir::dyn_cast<DeclareBarrierOp>(op)) {
            writer.createBarrier(barrierOp.barrier());
        } else if (mlir::dyn_cast<mlir::ReturnOp>(op) != nullptr || op == netFunc.getOperation()) {
            // do nothing
        } else {
            VPUX_THROW("Unknown Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        }
    };

    netFunc.walk(callback);

    std::vector<BlobWriter::TaskList> taskLists;
    taskLists.reserve(tasksMap.size());
    for (const auto& taskList : tasksMap) {
        log.trace("Serialize task list '{0}'", taskList.first);

        const auto serializedTaskList = writer.createVector(taskList.second);

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
