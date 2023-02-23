//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/EMU/graph-schema/export.hpp"

#include "vpux/compiler/dialect/EMU/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/generated/schema/gf_version.h"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

#include <precision_utils.h>
#include <transformations/utils/utils.hpp>
#include <version.hpp>

#include <unordered_map>

using namespace vpux;

namespace {

flatbuffers::Offset<MVCNN::Resources> createResources(EMU::BlobWriter& writer, mlir::ModuleOp module) {
    const EnumSet<VPU::ExecutorKind> supportedProcessors{
            VPU::ExecutorKind::SHAVE_UPA,  //
            VPU::ExecutorKind::SHAVE_NN,   //
            VPU::ExecutorKind::NCE,        //
            VPU::ExecutorKind::DPU         //
    };

    const auto usedMemory = writer.createVector(IE::getUsedMemory(module) | transformed([&](IE::MemoryResourceOp res) {
                                                    return VPUIP::createMemoryMapping(writer, res);
                                                }));

    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> executorsOffsets;
    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> processorVec;
    module.walk([&](IE::ExecutorResourceOp res) {
        if (const auto execKind = res.getKindAs<VPU::ExecutorKindAttr>()) {
            if (supportedProcessors.count(execKind.getValue()) != 0) {
                executorsOffsets.push_back(VPUIP::createProcessorMapping(writer, res, module));
                if (res.hasProcessorFrequency()) {
                    processorVec.push_back(VPUIP::createProcessorFreqMapping(writer, res));
                }
            }
        }
    });
    const auto executors = writer.createVector(executorsOffsets);
    const auto processorFrequency = writer.createVector(processorVec);

    SmallVector<flatbuffers::Offset<MVCNN::MemoryRelationshipMapping>> memoryVec;
    SmallVector<IE::MemoryResourceOp> memoryTypes;
    for (auto src : module.getOps<IE::MemoryResourceOp>()) {
        if (src->hasAttr(VPU::getMemoryBandwidthAttrName())) {
            memoryTypes.push_back(src);
        }
    }

    double DMA_BANDWIDTH = 20.0;
    for (auto src : memoryTypes) {
        for (auto dst : memoryTypes) {
            // TODO E#20897: update calculations with the below factors:
            // auto memoryBandwidth = VPUIP::getMemoryBandwidth(src);
            // auto memoryDerateFactor = VPUIP::getMemoryDerateFactor(src);
            if (src != dst) {
                memoryVec.push_back(VPUIP::createBandwidthMapping(writer, src, dst, DMA_BANDWIDTH));
            }
        }
    }

    const auto memoryBandwidthVec = writer.createVector(memoryVec);

    MVCNN::ResourcesBuilder builder(writer);
    builder.add_processor_allocation(executors);
    builder.add_memory_sizes(usedMemory);
    builder.add_processor_frequencies(processorFrequency);
    builder.add_memory_bandwidth(memoryBandwidthVec);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Version> createVersion(EMU::BlobWriter& writer, Logger log) {
    log.info("Blob version: majorV={0}, minorV={1}, patch={2}, hash={3}, context={4}", MVCNN_VERSION_MAJOR,
             MVCNN_VERSION_MINOR, MVCNN_VERSION_PATCH, VPUX_PLUGIN_VERSION, "VPUX Compiler");

    const auto serializedHash = writer.createString(VPUX_PLUGIN_VERSION);
    const auto serializedContext = writer.createString("VPUX Compiler");

    MVCNN::VersionBuilder builder(writer);
    builder.add_majorV(checked_cast<uint32_t>(MVCNN_VERSION_MAJOR));
    builder.add_minorV(checked_cast<uint32_t>(MVCNN_VERSION_MINOR));
    builder.add_patchV(checked_cast<uint32_t>(MVCNN_VERSION_PATCH));
    builder.add_hash(serializedHash);
    builder.add_context(serializedContext);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(
        EMU::BlobWriter& writer, mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::FuncOp netFunc,
        mlir::TimingScope& rootTiming, const std::vector<PreProcessInfo>& preprocessInfo,
        const std::vector<std::shared_ptr<const ov::Node>>& parameters,
        const std::vector<std::shared_ptr<const ov::Node>>& results, Logger log) {
    auto scopeTiming = rootTiming.nest("Create summary header");

    const auto allTasks = netFunc.getOps<EMU::SerializeInterface>();
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

        graphInputs.push_back(writer.createTensor(val, userInfo.name(), VPURT::BufferSection::NetworkInput));

        userInputs.push_back(writer.createTensor(userInfo.name(), userType, VPURT::BufferSection::NetworkInput));
    }

    SmallVector<EMU::BlobWriter::TensorReference> graphOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());

    auto returnOps = netFunc.getOps<mlir::ReturnOp>();
    VPUX_THROW_UNLESS(std::distance(returnOps.begin(), returnOps.end()) == 1,
                      "Only functions with 1 return are supported.");
    auto returnOperands = (*returnOps.begin()).getOperands();
    for (const auto& p : outputsInfo | indexed) {
        auto userInfo = p.value();

        auto val = returnOperands[p.index()];

        const auto userType = userInfo.userType().cast<mlir::ShapedType>();

        graphOutputs.push_back(writer.createTensor(val, userInfo.name(), VPURT::BufferSection::NetworkOutput));

        userOutputs.push_back(writer.createTensor(userInfo.name(), userType, VPURT::BufferSection::NetworkOutput));
    }

    auto createOVNodes = [&](const std::vector<std::shared_ptr<const ov::Node>>& nodes, const bool isResult) {
        SmallVector<VPUIP::BlobWriter::OVNodes> ovNodes;
        ovNodes.reserve(nodes.size());

        for (const auto& node : nodes) {
            VPUX_THROW_WHEN(node == nullptr, "Null OV node");
            const auto nodeFriendlyName = writer.createString(node->get_friendly_name());
            const auto nodeElementType = VPUIP::mapElementType.at(node->get_element_type());
            const auto nodeShape = writer.createVector(node->get_output_partial_shape(0).get_shape());
            const auto tmpTensorNames = node->get_output_tensor(0).get_names();
            SmallVector<VPUIP::BlobWriter::String> auxTensorNames;
            for (const auto& tensorName : tmpTensorNames) {
                auxTensorNames.push_back(writer.createString(tensorName));
            }
            const auto nodeTensorNames = writer.createVector(auxTensorNames);
            const auto tmpInputName =
                    isResult ? ngraph::op::util::create_ie_output_name(node->input_value(0)) : std::string("");
            const auto nodeInputName = writer.createString(tmpInputName);
            ovNodes.push_back(MVCNN::CreateOVNode(writer, nodeFriendlyName, nodeElementType, nodeShape, nodeTensorNames,
                                                  nodeInputName));
        }

        return ovNodes;
    };

    const auto ovParam = createOVNodes(parameters, false);
    const auto ovRes = createOVNodes(results, true);

    SmallVector<VPUIP::BlobWriter::PreprocessingInfo> preprocInfo;
    preprocInfo.reserve(preprocessInfo.size());

    for (const auto& pr : preprocessInfo) {
        preprocInfo.push_back(MVCNN::CreatepreprocessingInfo(writer, writer.createString(pr._inputName),
                                                             VPUIP::mapPreProcessColorFormat.at(pr._inputFormat),
                                                             VPUIP::mapPreProcessColorFormat.at(pr._outputFormat),
                                                             VPUIP::mapPreProcessResizeAlgorithm.at(pr._algorithm)));
    }

    const auto serializedVersion = createVersion(writer, log);
    const auto serializedName = writer.createString(module.getName().getValueOr("network"));
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources = createResources(writer, module);
    const auto serializedParameters = writer.createVector(ovParam);
    const auto serializedResults = writer.createVector(ovRes);
    const auto serializedPreProcInfo = writer.createVector(preprocInfo);

    MVCNN::SummaryHeaderBuilder builder(writer);
    builder.add_version(serializedVersion);
    builder.add_identifier(serializedName);
    builder.add_net_input(serializedGraphInputs);
    builder.add_net_output(serializedGraphOutputs);
    builder.add_task_count(checked_cast<uint32_t>(taskCount));
    builder.add_resources(serializedResources);
    builder.add_in_tensor_desc(serializedUserInputs);
    builder.add_out_tensor_desc(serializedUserOutputs);
    builder.add_ov_parameters(serializedParameters);
    builder.add_ov_results(serializedResults);
    builder.add_pre_process_info(serializedPreProcInfo);
    builder.add_device(VPUIP::mapTargetDevice(VPU::getArch(module)));
    builder.add_device_revision(VPUIP::mapTargetDeviceRevision(VPU::getArch(module)));
    return builder.Finish();
}

void serializeTensorDecls(EMU::BlobWriter& writer, mlir::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Serialize tensor declarations");

    size_t tempTensorInd = 0;
    netFunc.walk([&](EMU::SerializeInterface op) {
        for (auto result : op->getResults()) {
            auto users = result.getUsers();
            auto isNetworkResult = false;
            for (auto user : users) {
                if (mlir::isa<mlir::ReturnOp>(user))
                    isNetworkResult = true;
            }
            if (isNetworkResult)
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
                            VPURT::BufferSection::Constant, constTensorInd);
    }

    return binaryData;
}

SmallVector<EMU::BlobWriter::TaskList> serializeTaskLists(EMU::BlobWriter& writer, mlir::FuncOp netFunc,
                                                          mlir::TimingScope& rootTiming, Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize task lists");

    using TaskList = SmallVector<EMU::BlobWriter::Task>;
    TaskList tasks;

    netFunc.walk([&](EMU::SerializeInterface taskOp) {
        log.trace("Got '{0}' Task '{1}' at '{2}'", taskOp.getExecutorKind(), taskOp->getName(), taskOp->getLoc());
        tasks.push_back(writer.createTask(taskOp));
    });

    const auto serializedTaskList = writer.createVector(tasks);

    MVCNN::TaskListBuilder builder(writer);
    builder.add_content(serializedTaskList);

    SmallVector<EMU::BlobWriter::TaskList> taskLists = {builder.Finish()};

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

flatbuffers::DetachedBuffer vpux::EMU::exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                                    const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                                    const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                    const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                    Logger log) {
    log.setName("EMU::BackEnd");

    log.trace("Extract 'IE.{0}' from Module", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    EMU::BlobWriter writer(log.nest(), VPU::getArch(module));

    const auto header =
            createSummaryHeader(writer, module, netOp, netFunc, rootTiming, preprocessInfo, parameters, results, log);

    serializeTensorDecls(writer, netFunc, rootTiming);
    const auto binaryData = serializeBinaryData(writer, netFunc, rootTiming, log);
    const auto taskLists = serializeTaskLists(writer, netFunc, rootTiming, log);

    const auto graphFile = createGraphFile(writer, header, taskLists, binaryData, rootTiming);

    auto finalTiming = rootTiming.nest("Finalize serialized graph");
    writer.impl().Finish(graphFile, "BLOB");
    return writer.impl().Release();
}
