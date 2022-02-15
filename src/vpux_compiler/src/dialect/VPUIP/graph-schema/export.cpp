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

#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/generated/schema/gf_version.h"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <precision_utils.h>
#include <transformations/utils/utils.hpp>
#include <version.hpp>

#include <unordered_map>

// Base of frequency values used in tables (in MHz).
static constexpr uint32_t FREQ_BASE = 700;
// Step of frequency for each entry in tables (in MHz).
static constexpr uint32_t FREQ_STEP = 100;
// Base of bandwidth values used in tables (in MB/s).
static constexpr uint32_t BW_BASE = 2000;
// Step of bandwidth values used in tables (in MB/s).
static constexpr uint32_t BW_STEP = 100;
// Num entries in table, each entry contains set of values for particular frequency
static constexpr uint32_t NUM_ENTRIES = 5;

using namespace vpux;

namespace {

flatbuffers::Offset<MVCNN::Version> createVersion(VPUIP::BlobWriter& writer, Logger log) {
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

MVCNN::PhysicalProcessor createPhysicalProcessor(VPU::ExecutorKind execKind) {
    switch (execKind) {
    case VPU::ExecutorKind::SHAVE_UPA:
        return MVCNN::PhysicalProcessor_UPA_SHV;
    case VPU::ExecutorKind::SHAVE_NN:
        return MVCNN::PhysicalProcessor_NN_SHV;
    case VPU::ExecutorKind::NCE:
        return MVCNN::PhysicalProcessor_NCE_Cluster;
    case VPU::ExecutorKind::DPU:
        return MVCNN::PhysicalProcessor_NCE_PerClusterDPU;
    default:
        VPUX_THROW("Unsupported ExecutorKind '{0}'", execKind);
    }
}

void setActivityFactor(VPU::ExecutorKind execKind, MVCNN::ProcessorMappingBuilder& builder, mlir::ModuleOp module) {
    // TODO: calc this value during compilation
    static const float activityFactor = 90.0;
    const auto arch = VPU::getArch(module);
    if (arch == VPU::ArchKind::KMB || arch == VPU::ArchKind::TBH) {
        if (execKind == VPU::ExecutorKind::NCE || execKind == VPU::ExecutorKind::SHAVE_UPA) {
            builder.add_activity_factor(activityFactor);
        }
    } else if (arch == VPU::ArchKind::MTL) {
        if (execKind == VPU::ExecutorKind::NCE || execKind == VPU::ExecutorKind::SHAVE_NN) {
            builder.add_activity_factor(activityFactor);
        }
    }
}

flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorMapping(VPUIP::BlobWriter& writer,
                                                                    IE::ExecutorResourceOp res, mlir::ModuleOp module) {
    const auto execKindAttr = res.getKindAs<VPU::ExecutorKindAttr>();
    VPUX_THROW_UNLESS(execKindAttr != nullptr, "Got unknown executor kind '{0}'", res.getKind());

    const auto execKind = execKindAttr.getValue();
    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(execKind));
    builder.add_number(checked_cast<double>(res.count()));
    builder.add_is_bitmask(false);
    setActivityFactor(execKind, builder, module);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorFreqMapping(VPUIP::BlobWriter& writer,
                                                                        IE::ExecutorResourceOp res) {
    const auto execKindAttr = res.getKindAs<VPU::ExecutorKindAttr>();
    VPUX_THROW_UNLESS(execKindAttr != nullptr, "Got unknown executor kind '{0}'", res.getKind());

    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(execKindAttr.getValue()));
    builder.add_number(VPUIP::getProcessorFrequency(res));
    builder.add_is_bitmask(false);
    return builder.Finish();
}

MVCNN::PhysicalMem createPhysicalMem(VPU::MemoryKind mem) {
    switch (mem) {
    case VPU::MemoryKind::DDR:
        return MVCNN::PhysicalMem_DDR;
    case VPU::MemoryKind::CSRAM:
        return MVCNN::PhysicalMem_CSRAM;
    case VPU::MemoryKind::CMX_UPA:
        return MVCNN::PhysicalMem_UPA_CMX;
    case VPU::MemoryKind::CMX_NN:
        return MVCNN::PhysicalMem_NN_CMX;
    default:
        VPUX_THROW("Unsupported MemoryKind '{0}'", mem);
    }
}

flatbuffers::Offset<MVCNN::MemoryMapping> createMemoryMapping(VPUIP::BlobWriter& writer, IE::MemoryResourceOp res) {
    const auto memKindAttr = res.getKindAs<VPU::MemoryKindAttr>();

    MVCNN::MemoryMappingBuilder builder(writer);
    builder.add_item(createPhysicalMem(memKindAttr.getValue()));
    builder.add_number(checked_cast<double>(res.byteSize()));
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::MemoryRelationshipMapping> createBandwidthMapping(VPUIP::BlobWriter& writer,
                                                                             IE::MemoryResourceOp src,
                                                                             IE::MemoryResourceOp dst,
                                                                             double bandwidth) {
    MVCNN::MemoryRelationshipMappingBuilder builder(writer);
    const auto srcKind = src.getKindAs<VPU::MemoryKindAttr>();
    const auto dstKind = dst.getKindAs<VPU::MemoryKindAttr>();

    builder.add_from_item(createPhysicalMem(srcKind.getValue()));
    builder.add_to_item(createPhysicalMem(dstKind.getValue()));
    builder.add_number(bandwidth);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Resources> createResources(VPUIP::BlobWriter& writer, mlir::ModuleOp module) {
    const EnumSet<VPU::ExecutorKind> supportedProcessors{
            VPU::ExecutorKind::SHAVE_UPA,  //
            VPU::ExecutorKind::SHAVE_NN,   //
            VPU::ExecutorKind::NCE,        //
            VPU::ExecutorKind::DPU         //
    };

    const auto usedMemory = writer.createVector(IE::getUsedMemory(module) | transformed([&](IE::MemoryResourceOp res) {
                                                    return createMemoryMapping(writer, res);
                                                }));

    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> executorsOffsets;
    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> processorVec;
    module.walk([&](IE::ExecutorResourceOp res) {
        if (const auto execKind = res.getKindAs<VPU::ExecutorKindAttr>()) {
            if (supportedProcessors.count(execKind.getValue()) != 0) {
                executorsOffsets.push_back(createProcessorMapping(writer, res, module));
                if (res->hasAttr(VPU::getProcessorFrequencyAttrName())) {
                    processorVec.push_back(createProcessorFreqMapping(writer, res));
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
            // TODO EISW-20897: update calculations with the below factors:
            // auto memoryBandwidth = VPUIP::getMemoryBandwidth(src);
            // auto memoryDerateFactor = VPUIP::getMemoryDerateFactor(src);
            if (src != dst) {
                memoryVec.push_back(createBandwidthMapping(writer, src, dst, DMA_BANDWIDTH));
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

flatbuffers::Offset<MVCNN::PerformanceMetrics> createPerformanceMetrics(VPUIP::BlobWriter& writer) {
    // value in [0.0..1.0] range indicating scalability of network for a given DDR bandwidth.
    static const SmallVector<float> byBWScales({0.0F, 0.2F, 0.4F, 0.6F, 0.8F});

    // expected ticks (based on FRC @37.5MHz) an inference should take for a given DDR bandwidth.
    static const SmallVector<uint64_t> byBWTicks({10UL, 12UL, 14UL, 16UL, 18UL});

    SmallVector<flatbuffers::Offset<MVCNN::ScalabilityByBandwidth>> scaleByFreq;
    SmallVector<flatbuffers::Offset<MVCNN::InferenceTimingByBandwidth>> inferenceTimingByFreq;
    scaleByFreq.reserve(NUM_ENTRIES);
    inferenceTimingByFreq.reserve(NUM_ENTRIES);

    for (uint32_t i = 0; i < NUM_ENTRIES; ++i) {
        scaleByFreq.push_back(MVCNN::CreateScalabilityByBandwidth(writer, writer.createVector(byBWScales)));
        inferenceTimingByFreq.push_back(
                MVCNN::CreateInferenceTimingByBandwidth(writer, writer.createVector(byBWTicks)));
    }

    return MVCNN::CreatePerformanceMetrics(
            writer, FREQ_BASE, FREQ_STEP, BW_BASE, BW_STEP,
            MVCNN::CreateScalability(writer, writer.createVector(scaleByFreq)),
            MVCNN::CreateInferenceTiming(writer, writer.createVector(inferenceTimingByFreq)));
}

flatbuffers::Offset<MVCNN::ActKernelRuntime> createActKernelRuntime(VPUIP::BlobWriter& writer, mlir::ModuleOp module,
                                                                    mlir::FuncOp netFunc, Logger log) {
    // only SwKernelOp operations can generate kernelData
    auto graphHasKernels = false;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
            graphHasKernels = true;
        }
    });
    if (!graphHasKernels) {
        return {};
    }

    // TODO: always extract num shaves info from VPURT::SW.Runtime, which can be extracted from module
    constexpr auto maxShaves = 4;
    constexpr auto defaultStackSize = Byte(4_KB).count();
    constexpr auto alignmentReq = Byte(1_KB).count();

    auto swRuntimeOps = module.getOps<VPURT::SWRunTimeOp>();
    auto runtimeKernelDeclared = std::distance(swRuntimeOps.begin(), swRuntimeOps.end());
    VPUX_THROW_UNLESS(runtimeKernelDeclared <= 1, "if runtime kernel is present it should be unique, but found {0}",
                      runtimeKernelDeclared);

    flatbuffers::Offset<MVCNN::ActKernel> runtimeKernel;
    SmallVector<uint32_t> runtimeStacks(maxShaves, defaultStackSize);

    if (runtimeKernelDeclared) {
        auto swRuntimeOp = *swRuntimeOps.begin();
        runtimeKernel = writer.createRuntimeKernelTask(module, swRuntimeOp);
        runtimeStacks = parseIntArrayAttr<uint32_t>(swRuntimeOp.stacks());
    }

    SmallVector<flatbuffers::Offset<MVCNN::KernelDataReference>> stacks(runtimeStacks.size());

    const auto maxStackIt = std::max_element(runtimeStacks.begin(), runtimeStacks.end());
    const std::vector<uint8_t> shave_stack_data_max(*maxStackIt);

    for (auto shvInd : irange(runtimeStacks.size())) {
        const auto shaveStackData = makeArrayRef(shave_stack_data_max).take_front(runtimeStacks[shvInd]);

        log.trace("act-shave {0}_stack size is {1}", shvInd, shaveStackData.size());

        const auto dataName = llvm::formatv("actSHAVE{0}_stack", shvInd).str();
        stacks[shvInd] = writer.createKernelDataRef(dataName, 0, shaveStackData.size(), shaveStackData);
    }

    const auto stackBuffers = writer.createVector(stacks);

    VPUIP::BlobWriter::KernelDataRef scratchBuffer;
    if (!runtimeKernelDeclared) {
        constexpr auto scratchBufferSize = Byte(64_KB).count();

        const std::vector<uint8_t> scratchBufferData(alignmentReq + scratchBufferSize);
        constexpr uint64_t reservedOffset = 1;

        scratchBuffer =
                writer.createKernelDataRef("scratch_buffer", reservedOffset, scratchBufferSize, scratchBufferData);
    }

    MVCNN::ActKernelRuntimeBuilder builder(writer);
    builder.add_shaveStacks(stackBuffers);
    if (runtimeKernelDeclared) {
        builder.add_kernel(runtimeKernel);
    } else {
        builder.add_codeScratchBuffer(scratchBuffer);
    }

    return builder.Finish();
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(
        VPUIP::BlobWriter& writer, mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::FuncOp netFunc,
        bool withDynamicBarriers, mlir::TimingScope& rootTiming,
        const std::vector<vpux::PreProcessInfo>& preprocessInfo,
        const std::vector<std::shared_ptr<const ov::Node>>& parameters,
        const std::vector<std::shared_ptr<const ov::Node>>& results, Logger log) {
    auto scopeTiming = rootTiming.nest("Create summary header");

    const auto allTasks = netFunc.getOps<VPURT::TaskOp>();
    const auto allBarriers = netFunc.getOps<VPURT::ConfigureBarrierOp>();
    const auto taskCount =
            std::distance(allTasks.begin(), allTasks.end()) + std::distance(allBarriers.begin(), allBarriers.end());

    auto inputsInfo = netOp.getInputsInfo();
    auto outputsInfo = netOp.getOutputsInfo();
    auto profilingOutputsInfo = netOp.getProfilingOutputsInfo();

    SmallVector<VPUIP::BlobWriter::TensorReference> graphInputs, userInputs;
    graphInputs.reserve(inputsInfo.size());
    userInputs.reserve(inputsInfo.size());

    for (const auto& p : inputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        auto userInfo = p.value();
        const auto val = netFunc.getArgument(ind);

        const auto userType = userInfo.userType().cast<vpux::NDTypeInterface>();

        graphInputs.push_back(writer.createTensorRef(val, userInfo.name(), VPURT::BufferSection::NetworkInput, ind, 0));

        userInputs.push_back(
                writer.createTensorRef(userInfo.name(), userType, VPURT::BufferSection::NetworkInput, ind, 0));
    }

    SmallVector<VPUIP::BlobWriter::TensorReference> graphOutputs, graphProfilingOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());
    graphProfilingOutputs.reserve(profilingOutputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto ind = p.index();
        const auto funcArgInd = inputsInfo.size() + ind;

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto userType = userInfo.userType().cast<vpux::NDTypeInterface>();

        graphOutputs.push_back(
                writer.createTensorRef(val, userInfo.name(), VPURT::BufferSection::NetworkOutput, ind, 0));

        userOutputs.push_back(
                writer.createTensorRef(userInfo.name(), userType, VPURT::BufferSection::NetworkOutput, ind, 0));
    }

    for (const auto& p : profilingOutputsInfo | indexed) {
        const auto ind = p.index();
        const auto funcArgInd = inputsInfo.size() + outputsInfo.size() + ind;

        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        graphProfilingOutputs.push_back(
                writer.createTensorRef(val, p.value().name(), VPURT::BufferSection::ProfilingOutput, ind, 0));
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
            for (const auto tensorName : tmpTensorNames) {
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

    SmallVector<int8_t> options;
    if (withDynamicBarriers) {
        options.push_back(static_cast<int8_t>(MVCNN::ExecutionFlag_DynamicBarriers));
    }
    const auto serializedOptions = writer.createVector(options);

    const auto serializedVersion = createVersion(writer, log);
    const auto serializedName = writer.createString(module.getName().getValueOr("network"));
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedGraphProfilingOutputs = writer.createVector(graphProfilingOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources = createResources(writer, module);
    const auto serializedParameters = writer.createVector(ovParam);
    const auto serializedResults = writer.createVector(ovRes);
    const auto serializedPreProcInfo = writer.createVector(preprocInfo);
    const auto serializedActKernelsRuntime = createActKernelRuntime(writer, module, netFunc, log);
    const auto serializedPerformanceMetrics = createPerformanceMetrics(writer);

    MVCNN::SummaryHeaderBuilder builder(writer);
    builder.add_version(serializedVersion);
    builder.add_identifier(serializedName);
    builder.add_net_input(serializedGraphInputs);
    builder.add_net_output(serializedGraphOutputs);
    builder.add_profiling_output(serializedGraphProfilingOutputs);
    builder.add_task_count(checked_cast<uint32_t>(taskCount));
    builder.add_options(serializedOptions);
    builder.add_resources(serializedResources);
    builder.add_in_tensor_desc(serializedUserInputs);
    builder.add_out_tensor_desc(serializedUserOutputs);
    builder.add_ov_parameters(serializedParameters);
    builder.add_ov_results(serializedResults);
    builder.add_pre_process_info(serializedPreProcInfo);
    builder.add_device(VPUIP::mapTargetDevice(VPU::getArch(module)));
    builder.add_device_revision(VPUIP::mapTargetDeviceRevision(VPU::getArch(module)));
    builder.add_act_kernel_runtime(serializedActKernelsRuntime);
    builder.add_performance_metrics(serializedPerformanceMetrics);
    return builder.Finish();
}

void serializeTensorDecls(VPUIP::BlobWriter& writer, mlir::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Serialize tensor declarations");

    size_t tempTensorInd = 0;
    const auto createTensorRef = [&](VPURT::DeclareBufferOp bufOp, const Optional<int64_t> sparsityMapOffset = None,
                                     const Optional<int64_t> storageElementOffset = None) {
        auto sectionIndex = bufOp.getNonEmptySectionIndex();
        writer.createTensorRef(bufOp.buffer(), llvm::formatv("temp-{0}", tempTensorInd).str(), bufOp.section(),
                               sectionIndex, bufOp.byteOffset(), sparsityMapOffset, storageElementOffset);
        tempTensorInd++;
    };

    llvm::DenseSet<mlir::Operation*> sparseBuffers;
    netFunc.walk([&](VPURT::DeclareSparseBufferOp sparseBufOp) {
        Optional<int64_t> sparsityMapOffset = None;
        Optional<int64_t> storageElementOffset = None;
        if (sparseBufOp.sparsityMap()) {
            auto sparsityMapBufOp = sparseBufOp.sparsityMap().getDefiningOp<VPURT::DeclareBufferOp>();
            sparsityMapOffset = sparsityMapBufOp.byteOffset();
            createTensorRef(sparsityMapBufOp);
            sparseBuffers.insert(sparsityMapBufOp.getOperation());
        }
        if (sparseBufOp.storageElementTable()) {
            auto storageElementBufOp = sparseBufOp.storageElementTable().getDefiningOp<VPURT::DeclareBufferOp>();
            storageElementOffset = storageElementBufOp.byteOffset();
            createTensorRef(storageElementBufOp);
            sparseBuffers.insert(storageElementBufOp.getOperation());
        }

        auto bufOp = sparseBufOp.data().getDefiningOp<VPURT::DeclareBufferOp>();
        createTensorRef(bufOp, sparsityMapOffset, storageElementOffset);
        sparseBuffers.insert(bufOp.getOperation());
    });

    netFunc.walk([&](VPURT::DeclareBufferOp bufOp) {
        if (sparseBuffers.contains(bufOp.getOperation())) {
            return;
        }
        createTensorRef(bufOp);
    });
}

SmallVector<VPUIP::BlobWriter::BinaryData> serializeBinaryData(VPUIP::BlobWriter& writer, mlir::FuncOp netFunc,
                                                               mlir::TimingScope& rootTiming, Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize binary data");

    auto constOps = to_small_vector(netFunc.getOps<Const::DeclareOp>());

    SmallVector<std::vector<uint64_t>> bufs(constOps.size());

    loop_1d(LoopExecPolicy::Parallel, checked_cast<int64_t>(constOps.size()), [&](int64_t ind) {
        const auto attr = constOps[static_cast<size_t>(ind)].contentAttr();

        const auto type = attr.getType();
        const auto content = attr.fold();

        const auto totalByteSize = type.cast<vpux::NDTypeInterface>().getTotalAllocSize();
        bufs[static_cast<size_t>(ind)].resize(
                alignVal(static_cast<size_t>(totalByteSize.count()), sizeof(uint64_t)) / sizeof(uint64_t), 0);

        const auto buf = makeMutableArrayRef(reinterpret_cast<char*>(bufs[static_cast<size_t>(ind)].data()),
                                             totalByteSize.count());
        content.copyTo(buf);
    });

    SmallVector<VPUIP::BlobWriter::BinaryData> binaryData(constOps.size());

    for (auto constTensorInd : irange(constOps.size())) {
        auto constOp = constOps[constTensorInd];
        const auto& content = bufs[constTensorInd];

        log.trace("Got constant at '{0}' with type '{1}'", constOp->getLoc(), constOp.getType());

        binaryData[constTensorInd] = writer.createBinaryData(content, constOp.getType().cast<vpux::NDTypeInterface>());

        writer.createTensorRef(constOp.output(), llvm::formatv("constant-{0}", constTensorInd).str(),
                               VPURT::BufferSection::Constant, checked_cast<uint32_t>(constTensorInd), 0);
    }

    return binaryData;
}

SmallVector<VPUIP::BlobWriter::KernelData> serializeKernelData(VPUIP::BlobWriter& writer, mlir::FuncOp,
                                                               mlir::TimingScope&, Logger) {
    SmallVector<VPUIP::BlobWriter::KernelData> vec;
    for (auto&& e : writer.getKernelData()) {
        vec.push_back(e.second.data);
    }
    return vec;
}

SmallVector<VPUIP::BlobWriter::Barrier> serializeVirtBarriers(VPUIP::BlobWriter& writer, mlir::FuncOp netFunc,
                                                              bool withDynamicBarriers, mlir::TimingScope& rootTiming,
                                                              Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize virtual barriers");

    SmallVector<VPUIP::BlobWriter::Barrier> virtBarriers;

    netFunc.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        log.trace("Got virtual varrier at '{0}'", barrierOp->getLoc());

        VPUX_THROW_UNLESS(withDynamicBarriers, "Compiler was not configured for virtual barriers usage");

        const auto virtBarrier = writer.createBarrier(barrierOp.barrier());
        virtBarriers.push_back(virtBarrier);
    });

    return virtBarriers;
}

SmallVector<VPUIP::BlobWriter::TaskList> serializeTaskLists(VPUIP::BlobWriter& writer, mlir::FuncOp netFunc,
                                                            mlir::TimingScope& rootTiming, Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize task lists");

    using TaskList = SmallVector<VPUIP::BlobWriter::Task>;
    using TaskListMap = EnumMap<VPU::ExecutorKind, TaskList>;

    TaskList barriersList;
    netFunc.walk([&](VPURT::ConfigureBarrierOp barrierOp) {
        log.trace("Got '{0}' at '{1}'", barrierOp->getName(), barrierOp->getLoc());
        barriersList.push_back(writer.createTask(barrierOp));
    });

    TaskListMap tasksMap;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        log.trace("Got '{0}' Task '{1}' at '{2}'", taskOp.getExecutorKind(), taskOp->getName(), taskOp->getLoc());
        tasksMap[taskOp.getExecutorKind()].push_back(writer.createTask(taskOp));
    });

    SmallVector<VPUIP::BlobWriter::TaskList> taskLists;
    taskLists.reserve(tasksMap.size() + 1);

    const auto serializeTaskList = [&](const TaskList& taskList) {
        const auto serializedTaskList = writer.createVector(taskList);

        MVCNN::TaskListBuilder builder(writer);
        builder.add_content(serializedTaskList);
        taskLists.push_back(builder.Finish());
    };

    log.trace("Serialize barriers list");
    serializeTaskList(barriersList);

    for (const auto& taskList : tasksMap) {
        log.trace("Serialize tasks list '{0}'", taskList.first);
        serializeTaskList(taskList.second);
    }

    return taskLists;
}

flatbuffers::Offset<MVCNN::GraphFile> createGraphFile(VPUIP::BlobWriter& writer,
                                                      flatbuffers::Offset<MVCNN::SummaryHeader> header,
                                                      ArrayRef<VPUIP::BlobWriter::TaskList> taskLists,
                                                      ArrayRef<VPUIP::BlobWriter::BinaryData> binaryData,
                                                      ArrayRef<VPUIP::BlobWriter::KernelData> kernelData,
                                                      ArrayRef<VPUIP::BlobWriter::Barrier> virtBarriers,
                                                      mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Create graph file");

    const auto serializedTaskLists = writer.createVector(taskLists);
    const auto serializedBinaryData = writer.createVector(binaryData);
    const auto barrierTable = writer.createVector(virtBarriers);
    const auto serializedKernelData = writer.createVector(kernelData);

    MVCNN::GraphFileBuilder graphBuilder(writer);
    graphBuilder.add_header(header);
    graphBuilder.add_task_lists(serializedTaskLists);
    graphBuilder.add_binary_data(serializedBinaryData);
    graphBuilder.add_barrier_table(barrierTable);
    graphBuilder.add_kernel_data(serializedKernelData);

    return graphBuilder.Finish();
}

}  // namespace

flatbuffers::DetachedBuffer vpux::VPUIP::exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                                      const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                                      const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                      const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                      Logger log) {
    log.setName("VPUIP::BackEnd");

    log.trace("Extract 'IE.{0}' from Module", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    VPUIP::BlobWriter writer(log.nest());

    const auto withDynamicBarriers = !netFunc.getOps<VPURT::DeclareVirtualBarrierOp>().empty();

    const auto header = createSummaryHeader(writer, module, netOp, netFunc, withDynamicBarriers, rootTiming,
                                            preprocessInfo, parameters, results, log);

    serializeTensorDecls(writer, netFunc, rootTiming);
    const auto binaryData = serializeBinaryData(writer, netFunc, rootTiming, log);
    const auto virtBarriers = serializeVirtBarriers(writer, netFunc, withDynamicBarriers, rootTiming, log);
    const auto taskLists = serializeTaskLists(writer, netFunc, rootTiming, log);
    const auto kernelData = serializeKernelData(writer, netFunc, rootTiming, log);
    const auto graphFile = createGraphFile(writer, header, taskLists, binaryData, kernelData, virtBarriers, rootTiming);

    auto finalTiming = rootTiming.nest("Finalize serialized graph");
    writer.impl().Finish(graphFile, "BLOB");
    auto detached = writer.impl().Release();

    auto serializedGraphFile = MVCNN::GetGraphFile(detached.data());

    const uint64_t reserved_offset = 1;
    std::unordered_set<uint32_t> kernelDataAligned;

    // align KernelData section referenced by given KernelDataReference
    // returns moved offset
    auto alignKernelDataSection = [&](const MVCNN::KernelDataReference* section, auto sectionLogical) {
        //  current align requirements is 1KB, for .text, .data, .scratch
        constexpr auto alignmentReq = Byte(1_KB).count();

        auto section_data = serializedGraphFile->kernel_data()->Get(section->locale_offset())->data();

        // checking that current description designates aligned section -
        // TODO: implement cases where offset is 1 actually fixes alignment
        auto section_data_plus_offset = section_data->Data() + section->data_offset() - detached.data();

        if (!(section_data_plus_offset % alignmentReq)) {
            VPUX_THROW_UNLESS(section->data_offset() != reserved_offset,
                              "kernelDataReference: {0} {1}, offset from blob start: {2}, already aligned with "
                              "reserved data_offset={3}",
                              section->name()->c_str(), sectionLogical, section_data_plus_offset, reserved_offset);
            return static_cast<ptrdiff_t>(0);
        }
        ptrdiff_t offset = section_data->Data() - detached.data();
        log.trace("offset to kernel {0} {1} in Finished FBB is {2}", section->name()->c_str(), sectionLogical, offset);

        auto aligned_offset = llvm::alignTo(offset, alignmentReq);
        offset = aligned_offset - offset;

        // check whether given kernelData element already aligned
        if (kernelDataAligned.find(section->locale_offset()) == kernelDataAligned.end()) {
            // if there is no data - do not move
            if (section_data->Length() == 0) {
                return static_cast<ptrdiff_t>(0);
            }
            // check whether we do have a room for alignment
            VPUX_THROW_UNLESS(section_data->Length() > alignmentReq,
                              "cannot align section: {0} {1},  space(alignment + size): {2} alignment: {3}",
                              section->name()->c_str(), sectionLogical, section_data->Length(), alignmentReq);

            auto moveSize = section_data->Length() - alignmentReq;
            VPUX_THROW_UNLESS(section_data->Length() > moveSize + offset,
                              "cannot align section: {0} {1}, no room for moving space={2} offset={3} size={4}",
                              section->name()->c_str(), sectionLogical, section_data->Length(), offset, moveSize);

            log.trace("move kernel {0} {1} by {2} bytes to be {3}", section->name()->c_str(), sectionLogical, offset,
                      aligned_offset);

            memmove(const_cast<uint8_t*>(section_data->Data() + offset), section_data->Data(),
                    section_data->Length() - alignmentReq);

            // clear beginning
            memset(const_cast<uint8_t*>(section_data->Data()), 0, offset);
            // marking this kernel data content already aligned
            kernelDataAligned.insert(section->locale_offset());
        }

        return offset;
    };

    auto alignReferenceSection = [&](const MVCNN::KernelDataReference* section, uint64_t offset) {
        if (section->data_offset() != reserved_offset)
            return;
        // correcting data offset for section in schema
        auto table = reinterpret_cast<flatbuffers::Table*>(const_cast<MVCNN::KernelDataReference*>(section));

        // updating offset pointer
        // TODO: why we add here?
        table->SetField(MVCNN::KernelDataReference::VT_DATA_OFFSET,
                        checked_cast<uint32_t>(section->data_offset() + offset - reserved_offset), 0u);
    };

    auto alignSection = [&](const MVCNN::KernelDataReference* section, auto sectionLogical) {
        auto offset = alignKernelDataSection(section, sectionLogical);
        alignReferenceSection(section, offset);
    };

    // locating act-kernel
    for (auto&& task_list : *serializedGraphFile->task_lists()) {
        for (auto&& task : *task_list->content()) {
            if (auto actKernelTask = task->task_as_ActKernelTask()) {
                auto kernelTextSection = actKernelTask->kernel()->kernelText();
                alignSection(kernelTextSection, ".text");

                // Invocations aligning
                auto invocations = actKernelTask->invocations();
                for (auto&& invocation : *invocations) {
                    auto offset = alignKernelDataSection(invocation->dataSection(), ".data");
                    alignReferenceSection(invocation->dataSection(), offset);
                    alignReferenceSection(invocation->invocationArgs(), offset);
                }
            }
        }
    }

    if (serializedGraphFile->header()->act_kernel_runtime()) {
        // scratchBuffer aligning
        if (auto scratchBuffer = serializedGraphFile->header()->act_kernel_runtime()->codeScratchBuffer()) {
            alignSection(scratchBuffer, ".scratchBuffer");
        }
        // management kernel aligning if present
        if (auto managementKernel = serializedGraphFile->header()->act_kernel_runtime()->kernel()) {
            alignSection(managementKernel->kernelText(), ".runtime.text");
            alignSection(managementKernel->globalArgs(), ".runtime.data");
        }
    }

    return detached;
}
