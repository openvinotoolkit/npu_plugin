//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/backend/EMU.hpp"

#include "vpux/compiler/dialect/EMU/attributes/arch.hpp"
#include "vpux/compiler/dialect/EMU/blob_writer.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"

#include "vpux/utils/IE/loop.hpp"
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

flatbuffers::Offset<MVCNN::Version> createVersion(EMU::BlobWriter& writer, VPUIP::VersionAttr version) {
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

MVCNN::TargetDevice mapTargetDevice(const EMU::ArchKind kind) {
    switch (kind) {
    case EMU::ArchKind::KMB:
        return MVCNN::TargetDevice::TargetDevice_KMB;
    case EMU::ArchKind::TBH:
        return MVCNN::TargetDevice::TargetDevice_TBH;
    case EMU::ArchKind::MTL:
        return MVCNN::TargetDevice::TargetDevice_MTL;
    case EMU::ArchKind::LNL:
        return MVCNN::TargetDevice::TargetDevice_LNL;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::TargetDeviceRevision mapTargetDeviceRevision(const EMU::ArchKind kind) {
    switch (kind) {
    case EMU::ArchKind::KMB:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(EMU::BlobWriter& writer, mlir::ModuleOp module,
                                                              VPUIP::GraphOp graphOp, IE::CNNNetworkOp netOp,
                                                              mlir::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Create summary header");

    const auto allTasks = netFunc.getOps<EMU::TaskOpInterface>();
    const auto taskCount = std::distance(allTasks.begin(), allTasks.end());

    auto inputsInfo = netOp.getInputsInfo();
    auto outputsInfo = netOp.getOutputsInfo();

    SmallVector<EMU::BlobWriter::TensorReference> graphInputs, userInputs;
    graphInputs.reserve(inputsInfo.size());
    userInputs.reserve(inputsInfo.size());

    for (const auto& p : inputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(ind);

        const auto userType = userInfo.userType().cast<mlir::ShapedType>();

        graphInputs.push_back(writer.createTensor(val, userInfo.name(), EMU::MemoryLocation::ProgrammableInput));

        userInputs.push_back(writer.createTensor(userInfo.name(), userType, EMU::MemoryLocation::ProgrammableInput));
    }

    SmallVector<EMU::BlobWriter::TensorReference> graphOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto funcArgInd = inputsInfo.size() + p.index();

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto userType = userInfo.userType().cast<mlir::ShapedType>();

        graphOutputs.push_back(writer.createTensor(val, userInfo.name(), EMU::MemoryLocation::ProgrammableOutput));

        userOutputs.push_back(writer.createTensor(userInfo.name(), userType, EMU::MemoryLocation::ProgrammableOutput));
    }

    const auto serializedVersion = createVersion(writer, graphOp.version());
    const auto serializedName = writer.createString(module.getName().getValueOr("network"));
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);

    MVCNN::SummaryHeaderBuilder builder(writer);
    builder.add_version(serializedVersion);
    builder.add_identifier(serializedName);
    builder.add_net_input(serializedGraphInputs);
    builder.add_net_output(serializedGraphOutputs);
    builder.add_task_count(checked_cast<uint32_t>(taskCount));
    builder.add_in_tensor_desc(serializedUserInputs);
    builder.add_out_tensor_desc(serializedUserOutputs);
    builder.add_device(mapTargetDevice(EMU::getArch(module)));
    builder.add_device_revision(mapTargetDeviceRevision(EMU::getArch(module)));
    return builder.Finish();
}

void serializeTensorDecls(EMU::BlobWriter& writer, mlir::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Serialize tensor declarations");

    size_t tempTensorInd = 0;
    netFunc.walk([&](EMU::TaskOpInterface op) {
        for (auto result : op->getResults()) {
            if (result.isa<mlir::BlockArgument>())
                continue;
            writer.createTensor(result, llvm::formatv("temp-{0}", tempTensorInd).str());
        }

        ++tempTensorInd;
    });
}

SmallVector<EMU::BlobWriter::BinaryData> serializeBinaryData(EMU::BlobWriter& writer, mlir::FuncOp netFunc,
                                                             mlir::TimingScope& rootTiming, Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize binary data");

    auto constOps = to_small_vector(netFunc.getOps<Const::DeclareOp>());

    SmallVector<std::vector<uint64_t>> bufs(constOps.size());

    loop_1d(LoopExecPolicy::Parallel, checked_cast<int64_t>(constOps.size()), [&](int64_t ind) {
        const auto attr = constOps[static_cast<size_t>(ind)].contentAttr();

        const auto type = attr.getType();
        const auto content = attr.fold();

        const Byte elemTypeSize = getElemTypeSize(type);
        const size_t totalNumElements = type.getNumElements();
        const size_t totalByteSize = totalNumElements * elemTypeSize.count();

        bufs[static_cast<size_t>(ind)].resize(alignVal(totalByteSize, sizeof(uint64_t)) / sizeof(uint64_t), 0);

        const auto buf =
                makeMutableArrayRef(reinterpret_cast<char*>(bufs[static_cast<size_t>(ind)].data()), totalByteSize);
        content.copyTo(buf);
    });

    SmallVector<EMU::BlobWriter::BinaryData> binaryData(constOps.size());

    for (auto constTensorInd : irange(constOps.size())) {
        auto constOp = constOps[constTensorInd];
        const auto& content = bufs[constTensorInd];

        log.trace("Got constant at '{0}' with type '{1}'", constOp->getLoc(), constOp.getType());

        binaryData[constTensorInd] = writer.createBinaryData(content, constOp.getType().cast<mlir::ShapedType>());

        writer.createTensor(constOp.output(), llvm::formatv("constant-{0}", constTensorInd).str(),
                            EMU::MemoryLocation::GraphFile);
    }

    return binaryData;
}

SmallVector<EMU::BlobWriter::TaskList> serializeTaskLists(EMU::BlobWriter& writer, mlir::FuncOp netFunc,
                                                          mlir::TimingScope& rootTiming, Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize task lists");

    using TaskList = SmallVector<EMU::BlobWriter::Task>;
    using TaskListMap = EnumMap<EMU::TaskType, TaskList>;
    TaskListMap tasksMap;

    netFunc.walk([&](EMU::TaskOpInterface taskOp) {
        log.trace("Got '{0}' Task '{1}' at '{2}'", taskOp.getTaskType(), taskOp->getName(), taskOp->getLoc());
        tasksMap[taskOp.getTaskType()].push_back(writer.createTask(taskOp));
    });

    SmallVector<EMU::BlobWriter::TaskList> taskLists;
    taskLists.reserve(tasksMap.size());

    for (const auto& taskList : tasksMap) {
        log.trace("Serialize task list '{0}'", taskList.first);

        const auto serializedTaskList = writer.createVector(taskList.second);

        MVCNN::TaskListBuilder builder(writer);
        builder.add_content(serializedTaskList);
        taskLists.push_back(builder.Finish());
    }

    return taskLists;
}

flatbuffers::Offset<MVCNN::GraphFile> createGraphFile(EMU::BlobWriter& writer,
                                                      flatbuffers::Offset<MVCNN::SummaryHeader> header,
                                                      ArrayRef<EMU::BlobWriter::TaskList> taskLists,
                                                      ArrayRef<EMU::BlobWriter::BinaryData> binaryData,
                                                      mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Create graph file");

    const auto serializedTaskLists = writer.createVector(taskLists);
    const auto serializedBinaryData = writer.createVector(binaryData);

    MVCNN::GraphFileBuilder graphBuilder(writer);
    graphBuilder.add_header(header);
    graphBuilder.add_task_lists(serializedTaskLists);
    graphBuilder.add_binary_data(serializedBinaryData);

    return graphBuilder.Finish();
}

}  // namespace

flatbuffers::DetachedBuffer vpux::EMU::exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming, Logger log) {
    log.setName("EMU::BackEnd");

    log.trace("Extract 'IE.{0}' from Module", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Extract 'EMU.{0}' from Module", VPUIP::GraphOp::getOperationName());
    auto graphOp = VPUIP::GraphOp::getFromModule(module);

    EMU::BlobWriter writer(log.nest());

    const auto header = createSummaryHeader(writer, module, graphOp, netOp, netFunc, rootTiming);

    serializeTensorDecls(writer, netFunc, rootTiming);
    const auto binaryData = serializeBinaryData(writer, netFunc, rootTiming, log);
    const auto taskLists = serializeTaskLists(writer, netFunc, rootTiming, log);

    const auto graphFile = createGraphFile(writer, header, taskLists, binaryData, rootTiming);

    auto finalTiming = rootTiming.nest("Finalize serialized graph");
    writer.impl().Finish(graphFile, "BLOB");
    return writer.impl().Release();
}
