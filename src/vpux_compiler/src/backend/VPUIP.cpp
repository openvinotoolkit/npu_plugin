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

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/core/layers.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/generated/schema/gf_version.h"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/elf_blob_serializer.hpp"

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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <flatbuffers/flatbuffers.h>

#include <precision_utils.h>
#include <version.hpp>

#include <deque>
#include <unordered_map>

#include "host_parsed_inference.h"
#include "nce_2p7_hw.h"

#include "vpux/compiler/utils/quantization.hpp"

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
                                                                    IERT::ExecutorResourceOp res,
                                                                    mlir::ModuleOp module) {
    const auto execKindAttr = res.kind().dyn_cast_or_null<VPU::ExecutorKindAttr>();
    VPUX_THROW_UNLESS(execKindAttr != nullptr, "Got unknown executor kind '{0}'", res.kind());

    const auto execKind = execKindAttr.getValue();
    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(execKind));
    builder.add_number(checked_cast<double>(res.count()));
    builder.add_is_bitmask(false);
    setActivityFactor(execKind, builder, module);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorFreqMapping(VPUIP::BlobWriter& writer,
                                                                        IERT::ExecutorResourceOp res) {
    const auto execKindAttr = res.kind().dyn_cast_or_null<VPU::ExecutorKindAttr>();
    VPUX_THROW_UNLESS(execKindAttr != nullptr, "Got unknown executor kind '{0}'", res.kind());

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

flatbuffers::Offset<MVCNN::MemoryMapping> createMemoryMapping(VPUIP::BlobWriter& writer, IERT::MemoryResourceOp res) {
    const auto memKindAttr = res.kindAttr().dyn_cast_or_null<VPU::MemoryKindAttr>();
    VPUX_THROW_UNLESS(memKindAttr != nullptr, "Got unknown memory space kind '{0}'", res.kindAttr());

    MVCNN::MemoryMappingBuilder builder(writer);
    builder.add_item(createPhysicalMem(memKindAttr.getValue()));
    builder.add_number(checked_cast<double>(res.byteSize()));
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::MemoryRelationshipMapping> createBandwidthMapping(VPUIP::BlobWriter& writer,
                                                                             IERT::MemoryResourceOp src,
                                                                             IERT::MemoryResourceOp dst,
                                                                             double bandwidth) {
    MVCNN::MemoryRelationshipMappingBuilder builder(writer);
    const auto srcKind = src.kindAttr().dyn_cast_or_null<VPU::MemoryKindAttr>();
    VPUX_THROW_UNLESS(srcKind != nullptr, "Got unknown memory space kind '{0}'", src.kindAttr());
    const auto dstKind = dst.kindAttr().dyn_cast_or_null<VPU::MemoryKindAttr>();
    VPUX_THROW_UNLESS(dstKind != nullptr, "Got unknown memory space kind '{0}'", dst.kindAttr());
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

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    VPUX_THROW_UNLESS(resources != nullptr, "Missing IERT run-time resources information");

    const auto usedMemory =
            writer.createVector(resources.getUsedMemory() | transformed([&](IERT::MemoryResourceOp res) {
                                    return createMemoryMapping(writer, res);
                                }));

    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> executorsOffsets;
    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> processorVec;
    resources.walk([&](IERT::ExecutorResourceOp res) {
        if (const auto execKind = res.kind().dyn_cast<VPU::ExecutorKindAttr>()) {
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
    SmallVector<IERT::MemoryResourceOp> memoryTypes;
    resources.walk([&](IERT::MemoryResourceOp src) {
        if (src->hasAttr(VPU::getMemoryBandwidthAttrName())) {
            memoryTypes.push_back(src);
        }
    });

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

flatbuffers::Offset<MVCNN::ActKernelRuntime> createActKernelRuntime(VPUIP::BlobWriter& writer,
                                                                    mlir::ModuleOp /*module*/, mlir::FuncOp netFunc,
                                                                    Logger log) {
    // only SwKernelOp operations can generate kernelData, from either built-in functions or from custom
    auto graphHasKernels = false;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getTaskType() == vpux::VPUIP::TaskType::ACTShave) {
            graphHasKernels = true;
        }
    });
    if (!graphHasKernels) {
        return {};
    }

    // TODO: extract num shaves info from IERT::RuntimeResourcesOp, which can be extracted from module
    const long int maxShaves = 4;

    const auto stack_size{1U << 12};  // 4KB stack

    SmallVector<uint8_t, stack_size> shave_stack_data(stack_size);
    std::vector<flatbuffers::Offset<MVCNN::KernelDataReference>> stacks(maxShaves);  // 4 Activation SHAVEs for MTL

    for (uint32_t shv{}; shv < maxShaves; ++shv) {
        log.trace("act-shave {0}_stack size is {1}", shv, stack_size);

        stacks[shv] = writer.createKernelDataRef("actSHAVE" + std::to_string(shv) + "_stack",
                                                 vpux::VPUIP::MemoryLocation::GFEmbeddedKernel, 0, stack_size,
                                                 shave_stack_data);
    }

    const auto stackBuffers = writer.createVector(stacks);

    SmallVector<uint8_t, 1024 + (1U << 16)> scratch_buffer(1024 + (1U << 16));  // 64KB scratch buffer + 1024 to align

    const uint64_t non_empty_offset = 1;
    const auto scratchBuffer =
            writer.createKernelDataRef("scratch_buffer", vpux::VPUIP::MemoryLocation::GFEmbeddedKernel,
                                       non_empty_offset, scratch_buffer.size() - 1024, scratch_buffer);

    MVCNN::ActKernelRuntimeBuilder builder(writer);
    builder.add_shaveStacks(stackBuffers);
    builder.add_codeScratchBuffer(scratchBuffer);

    return builder.Finish();
}

MVCNN::TargetDevice mapTargetDevice(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::KMB:
        return MVCNN::TargetDevice::TargetDevice_KMB;
    case VPU::ArchKind::TBH:
        return MVCNN::TargetDevice::TargetDevice_TBH;
    case VPU::ArchKind::MTL:
        return MVCNN::TargetDevice::TargetDevice_MTL;
    case VPU::ArchKind::LNL:
        return MVCNN::TargetDevice::TargetDevice_LNL;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::TargetDeviceRevision mapTargetDeviceRevision(VPU::ArchKind kind) {
    switch (kind) {
    case VPU::ArchKind::KMB:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }
}

const EnumMap<vpux::PreProcessColorSpace, MVCNN::PreProcessColorSpace> mapPreProcessColorFormat = {
        {vpux::PreProcessColorSpace::BGR, MVCNN::PreProcessColorSpace::PreProcessColorSpace_BGR},
        {vpux::PreProcessColorSpace::RGB, MVCNN::PreProcessColorSpace::PreProcessColorSpace_RGB},
        {vpux::PreProcessColorSpace::NV12, MVCNN::PreProcessColorSpace::PreProcessColorSpace_NV12},
        {vpux::PreProcessColorSpace::I420, MVCNN::PreProcessColorSpace::PreProcessColorSpace_I420},
        {vpux::PreProcessColorSpace::NONE, MVCNN::PreProcessColorSpace::PreProcessColorSpace_DEFAULT},
};

const EnumMap<vpux::PreProcessResizeAlgorithm, MVCNN::PreProcessResizeAlgorithm> mapPreProcessResizeAlgorithm = {
        {vpux::PreProcessResizeAlgorithm::RESIZE_BILINEAR,
         MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_BILINEAR},
        {vpux::PreProcessResizeAlgorithm::RESIZE_AREA,
         MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_AREA},
        {vpux::PreProcessResizeAlgorithm::NO_RESIZE,
         MVCNN::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_NO_RESIZE},
};

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(VPUIP::BlobWriter& writer, mlir::ModuleOp module,
                                                              IE::CNNNetworkOp netOp, mlir::FuncOp netFunc,
                                                              bool withDynamicBarriers, mlir::TimingScope& rootTiming,
                                                              const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                                              Logger log) {
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

        const auto userType = userInfo.userType().cast<mlir::ShapedType>();

        graphInputs.push_back(
                writer.createTensor(val, userInfo.name(), VPUIP::MemoryLocation::ProgrammableInput, ind, 0));

        userInputs.push_back(
                writer.createTensor(userInfo.name(), userType, VPUIP::MemoryLocation::ProgrammableInput, ind, 0));
    }

    SmallVector<VPUIP::BlobWriter::TensorReference> graphOutputs, graphProfilingOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());
    graphProfilingOutputs.reserve(profilingOutputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        const auto funcArgInd = inputsInfo.size() + p.index();

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto userType = userInfo.userType().cast<mlir::ShapedType>();

        graphOutputs.push_back(
                writer.createTensor(val, userInfo.name(), VPUIP::MemoryLocation::ProgrammableOutput, ind, 0));

        userOutputs.push_back(
                writer.createTensor(userInfo.name(), userType, VPUIP::MemoryLocation::ProgrammableOutput, ind, 0));
    }

    for (const auto& p : profilingOutputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        const auto funcArgInd = inputsInfo.size() + outputsInfo.size() + p.index();

        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        graphProfilingOutputs.push_back(
                writer.createTensor(val, p.value().name(), VPUIP::MemoryLocation::ProfilingOutput, ind, 0));
    }

    SmallVector<VPUIP::BlobWriter::PreprocessingInfo> preprocInfo;
    preprocInfo.reserve(preprocessInfo.size());

    for (const auto& pr : preprocessInfo) {
        preprocInfo.push_back(MVCNN::CreatepreprocessingInfo(
                writer, writer.createString(pr._inputName), mapPreProcessColorFormat.at(pr._inputFormat),
                mapPreProcessColorFormat.at(pr._outputFormat), mapPreProcessResizeAlgorithm.at(pr._algorithm)));
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
    const auto serializedPreProcInfo = writer.createVector(preprocInfo);
    const auto serializedActKernelsRuntime = createActKernelRuntime(writer, module, netFunc, log);

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
    builder.add_pre_process_info(serializedPreProcInfo);
    builder.add_device(mapTargetDevice(VPU::getArch(module)));
    builder.add_device_revision(mapTargetDeviceRevision(VPU::getArch(module)));
    builder.add_act_kernel_runtime(serializedActKernelsRuntime);
    return builder.Finish();
}

void serializeTensorDecls(VPUIP::BlobWriter& writer, mlir::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Serialize tensor declarations");

    size_t tempTensorInd = 0;
    netFunc.walk([&](VPURT::DeclareBufferOp tensorOp) {
        writer.createTensor(tensorOp.memory(), llvm::formatv("temp-{0}", tempTensorInd).str(), tensorOp.locale(),
                            parseIntArrayAttr<uint32_t>(tensorOp.localeIndex()), tensorOp.dataIndex(),
                            tensorOp.sparsityIndex(), tensorOp.storageElementIndex(), tensorOp.storageElementSize(),
                            tensorOp.leadingOffset(), tensorOp.trailingOffset());

        ++tempTensorInd;
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

        const Byte elemTypeSize = getElemTypeSize(type);
        const size_t totalNumElements = type.getNumElements();
        const size_t totalByteSize = totalNumElements * elemTypeSize.count();

        bufs[static_cast<size_t>(ind)].resize(alignVal(totalByteSize, sizeof(uint64_t)) / sizeof(uint64_t), 0);

        const auto buf =
                makeMutableArrayRef(reinterpret_cast<char*>(bufs[static_cast<size_t>(ind)].data()), totalByteSize);
        content.copyTo(buf);
    });

    SmallVector<VPUIP::BlobWriter::BinaryData> binaryData(constOps.size());

    for (auto constTensorInd : irange(constOps.size())) {
        auto constOp = constOps[constTensorInd];
        const auto& content = bufs[constTensorInd];

        log.trace("Got constant at '{0}' with type '{1}'", constOp->getLoc(), constOp.getType());

        binaryData[constTensorInd] = writer.createBinaryData(content, constOp.getType().cast<mlir::ShapedType>());

        writer.createTensor(constOp.output(), llvm::formatv("constant-{0}", constTensorInd).str(),
                            VPUIP::MemoryLocation::GraphFile, checked_cast<uint32_t>(constTensorInd), 0);
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
    using TaskListMap = EnumMap<VPUIP::TaskType, TaskList>;
    TaskListMap tasksMap;

    netFunc.walk([&](VPURT::ConfigureBarrierOp taskOp) {
        log.trace("Got '{0}' Task '{1}' at '{2}'", taskOp.getTaskType(), taskOp->getName(), taskOp->getLoc());
        tasksMap[taskOp.getTaskType()].push_back(writer.createTask(taskOp));
    });

    netFunc.walk([&](VPURT::TaskOp taskOp) {
        log.trace("Got '{0}' Task '{1}' at '{2}'", taskOp.getTaskType(), taskOp->getName(), taskOp->getLoc());
        tasksMap[taskOp.getTaskType()].push_back(writer.createTask(taskOp));
    });

    SmallVector<VPUIP::BlobWriter::TaskList> taskLists;
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
                                                      Logger log) {
    log.setName("VPUIP::BackEnd (Graph File)");

    log.trace("Extract 'IE.{0}' from Module (Graph File)", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    VPUIP::BlobWriter writer(log.nest());

    const auto withDynamicBarriers = !netFunc.getOps<VPURT::DeclareVirtualBarrierOp>().empty();

    const auto header =
            createSummaryHeader(writer, module, netOp, netFunc, withDynamicBarriers, rootTiming, preprocessInfo, log);

    serializeTensorDecls(writer, netFunc, rootTiming);
    const auto binaryData = serializeBinaryData(writer, netFunc, rootTiming, log);
    const auto virtBarriers = serializeVirtBarriers(writer, netFunc, withDynamicBarriers, rootTiming, log);
    const auto taskLists = serializeTaskLists(writer, netFunc, rootTiming, log);
    const auto kernelData = serializeKernelData(writer, netFunc, rootTiming, log);
    const auto graphFile = createGraphFile(writer, header, taskLists, binaryData, kernelData, virtBarriers, rootTiming);

    auto finalTiming = rootTiming.nest("Finalize serialized graph (Graph File)");
    writer.impl().Finish(graphFile, "BLOB");
    auto detached = writer.impl().Release();

    auto serializedGraphFile = MVCNN::GetGraphFile(detached.data());

    const uint64_t reserved_offset = 1;
    std::unordered_set<uint32_t> kernelDataAligned;

    // align KernelData section referenced by given KernelDataReference
    // returns moved offset
    auto alignKernelDataSection = [&](const MVCNN::KernelDataReference* section, auto sectionLogical) {
        //  current align requirements is 1KB, for .text, .data, .scratch
        constexpr uint16_t alignmentReq = 1 * 1024;

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

    // scratchBuffer aligning
    if (serializedGraphFile->header()->act_kernel_runtime()) {
        auto scratchBuffer = serializedGraphFile->header()->act_kernel_runtime()->codeScratchBuffer();
        alignSection(scratchBuffer, ".scratchBuffer");
    }

    return detached;
}

constexpr auto KERNEL_SIZE_MIN = 1;
constexpr auto KERNEL_SIZE_MAX = 11;
constexpr auto KERNEL_STRIDE_MIN = 1;
constexpr auto KERNEL_STRIDE_MAX = 8;

host_parsing::DType Type2DType(mlir::Type type) {
    if (type.isF64()) {
        return host_parsing::DType::FP64;
    } else if (type.isF32()) {
        return host_parsing::DType::FP32;
    } else if (type.isF16()) {
        return host_parsing::DType::FP16;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(uint64_t))) {
        return host_parsing::DType::U64;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(uint32_t))) {
        return host_parsing::DType::U32;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(uint16_t))) {
        return host_parsing::DType::U16;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(uint8_t))) {
        return host_parsing::DType::U8;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int64_t))) {
        return host_parsing::DType::I64;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return host_parsing::DType::I32;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int16_t))) {
        return host_parsing::DType::I16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return host_parsing::DType::I8;
    } else if (type.isSignedInteger(4)) {
        return host_parsing::DType::I4;
    } else if (type.isSignedInteger(2)) {
        // OPEN: is there U2 ?
        return host_parsing::DType::I2;
    } else if (type.isInteger(1)) {
        return host_parsing::DType::BIN;
    } else if (type.isBF16()) {
        return host_parsing::DType::BFP16;
    } else if (type.isUnsignedInteger(4)) {
        return host_parsing::DType::U4;
    }

    // OPEN: FP8, I4X, LOG, I2X
    VPUX_THROW("Unsupported data type {0}", type);
}

host_parsing::InputTensorDType Type2InputDType(mlir::Type type) {
    if (type.isF16()) {
        return host_parsing::InputTensorDType::FP16;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(uint8_t))) {
        return host_parsing::InputTensorDType::U8;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return host_parsing::InputTensorDType::I8;
    } else if (type.isSignedInteger(4)) {
        return host_parsing::InputTensorDType::I4;
    } else if (type.isSignedInteger(2)) {
        // OPEN: is there U2 ?
        return host_parsing::InputTensorDType::I2;
    } else if (type.isBF16()) {
        return host_parsing::InputTensorDType::BF16;
    } else if (type.isUnsignedInteger(4)) {
        return host_parsing::InputTensorDType::U4;
    } else if (type.isInteger(1)) {
        return host_parsing::InputTensorDType::BIN;
    }

    // OPEN: FP8 is not supported yet

    VPUX_THROW("Encountered unsupported data type {0}", type);
}

host_parsing::MpeActivationWeightDtype Type2WeightsDType(mlir::Type type) {
    if (type.isF16()) {
        return host_parsing::MpeActivationWeightDtype::FP16;
    } else if (type.isUnsignedInteger(CHAR_BIT * sizeof(uint8_t))) {
        return host_parsing::MpeActivationWeightDtype::U8;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return host_parsing::MpeActivationWeightDtype::I8;
    } else if (type.isSignedInteger(4)) {
        return host_parsing::MpeActivationWeightDtype::I4;
    } else if (type.isSignedInteger(2)) {
        // OPEN: is there U2 ?
        return host_parsing::MpeActivationWeightDtype::I2;
    } else if (type.isInteger(1)) {
        return host_parsing::MpeActivationWeightDtype::BIN;
    }

    // OPEN: I4X, I2X are not supported

    VPUX_THROW("Encountered unsupported data type {0}", type);
}

host_parsing::OutputTensorDType Type2OutputDType(mlir::Type type) {
    if (type.isF16()) {
        return host_parsing::OutputTensorDType::FP16;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return host_parsing::OutputTensorDType::I8;
    } else if (type.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return host_parsing::OutputTensorDType::I32;
    } else if (type.isSignedInteger(4)) {
        return host_parsing::OutputTensorDType::I4;
    } else if (type.isUnsignedInteger(2)) {
        // OPEN: is there U2 ?
        return host_parsing::OutputTensorDType::I2;
    } else if (type.isInteger(1)) {
        return host_parsing::OutputTensorDType::BIN;
    } else if (type.isF32()) {
        return host_parsing::OutputTensorDType::FP32;
    } else if (type.isBF16()) {
        return host_parsing::OutputTensorDType::BF16;
    }

    // OPEN: U8F, G8, LOG are not supported

    VPUX_THROW("Encountered unsupported data type {0}", type);
}

bool areSupportedInputOutputTypes(host_parsing::DType in_type, host_parsing::DType out_type) {
    bool in_supported = (in_type == host_parsing::DType::BFP16) || (in_type == host_parsing::DType::FP8) || (in_type == host_parsing::DType::U8) ||
                        (in_type == host_parsing::DType::I8) || (in_type == host_parsing::DType::I4) || (in_type == host_parsing::DType::FP16);

    bool out_supported = (out_type == host_parsing::DType::BFP16) || (out_type == host_parsing::DType::FP8) || (out_type == host_parsing::DType::U8) ||
                         (out_type == host_parsing::DType::I8) || (out_type == host_parsing::DType::I32) || (out_type == host_parsing::DType::I4) ||
                         (out_type == host_parsing::DType::FP16) || (out_type == host_parsing::DType::FP32);


    return in_supported && out_supported;
}

// PPE activation function choices in 2p7
enum class ActivationFunction { no_activation_function, relu, relu_x, leaky_relu, unsupported };

using u32f32 = union {
    uint32_t u32;
    float f32;
};

struct ActivationFunctionDesc {
    float alpha;
    u32f32 alphaFP32;
    uint32_t alphaMult;  // Mult Register value
    uint32_t alphaShift; // Shift register value (number of bits to shift left by)
    ActivationFunction funcType;
    int32_t clampLow;
    int32_t clampHigh;

    ActivationFunctionDesc()
        : alpha(1.0)
        , alphaMult(0)
        , alphaShift(1)
        , funcType(ActivationFunction::no_activation_function)
        , clampLow(0)
        , clampHigh(0) {
        alphaFP32.u32 = 0;
    }
};

#define PACK_F32(x, y, z)     ((x << 31) + (y << 23) + z)
#define PACK_F16(x, y, z)     ((x << 15) + (y << 10) + z)
#define PACK_B16(x, y, z)     ((x << 15) + (y << 7) + (z))

#define EXTRACT_F32_SIGN(x)   ((x >> 31) & 0x1)
#define EXTRACT_F32_EXP(x)    ((x >> 23) & 0xFF)
#define EXTRACT_F32_FRAC(x)   (x & 0x007FFFFF)
#define PACK_B16(x, y, z) ((x << 15) + (y << 7) + (z))

#define F32_EX_INEXACT 0x00000001    // 0x00000020
#define F32_EX_UNDERFLOW 0x00000008  // 0x00000010
#define F32_EX_INVALID     0x00000004//0x00000001
#define F32_EX_UNDERFLOW   0x00000008//0x00000010
#define F32_EX_OVERFLOW    0x00000010//0x00000008

#define F32_RND_NEAREST_EVEN 0
#define F32_RND_MINUS_INF 1
#define F32_RND_PLUS_INF 2
#define F32_RND_TO_ZERO 3

// Constants for converting to BF16
constexpr uint32_t fp32FracBits = 23;
constexpr uint32_t fp16ExpBias = 15;
constexpr uint32_t fp16FracBits = 10;
constexpr uint32_t bf16FracBits = 7;
constexpr uint32_t bf16NanOutput = 0x7FC0;
constexpr uint32_t fp32NanOutput = 0x7FC00000; // Aligns with Synopsys DWC_FP_MULT fixed NAN output

// Apply RNE rounding to fp16 fractional part
// Note if overflow of frac (>10 bits) occurs when rounding up this will
// propagate to the exponent when added in the PACK16 function
uint32_t RoundFp16(uint32_t &dataIn, uint32_t fracWidth) {
    uint32_t frac;
    uint32_t precisionBitsMask;
    uint32_t precisionBits; // Bits used to determine precision
    uint32_t tie;

    if (fracWidth > fp16FracBits) {
        precisionBitsMask = (0x01 << (fracWidth - fp16FracBits)) - 1; // Bits to determine rounding
        precisionBits = dataIn & precisionBitsMask;
        frac = dataIn >> (fracWidth - fp16FracBits); // Pre-rounded fp16 fraction

        tie = 0x01 << (fracWidth - fp16FracBits - 1); // -1 so that we end up with leading 1-bit at MSB of precisionBits
        if (precisionBits > tie) {
            frac++;
        } else if (precisionBits == tie) {
            if ((frac & 0x01)) {
                frac++; // Add 1 if tie and frac is odd (ties to even)
            }
        }
    } else {
        precisionBits = 0;                           // No rounding needed
        frac = dataIn << (fp16FracBits - fracWidth); // fp16 fraction
    }

    return frac;
}

// Taken from ppeQuantisation.cpp in vpu_sysc:
//
// Convert the signed fixed point number to FP16
// Fixed point number is in 2's complement format so need to convert to ((-1)^S)*(1.x1x2...x9)*2^E format
// where: S is the sign
//        x1x2...x9 are the fractional bits after the leading 1-bit
//        E is the biased exponent
int32_t fixedPointToFp16(int32_t x, uint32_t intBits, uint32_t fracBits) {
    uint32_t result;
    uint32_t sign;
    int32_t exp;
    uint32_t frac;

    // Extract the sign and absolute value of x
    sign = (x >> (intBits + fracBits - 1)) & 0x01; // Extract sign bit (assumes signed fixed point input)
    uint32_t xAbs;
    if (sign) {
        xAbs = (~x + 1);
    } else {
        xAbs = x;
    }

    // Detect position of leading 1-bit of input (excluding the sign)
    uint32_t xAbsShift = xAbs;
    uint32_t count = 0;
    while (xAbsShift >>= 1) // Shift right until the leading 1 bit has been shifted off (xAbs becomes false)
    {
        count++;
    }

    // Calculate the fp16 exponent
    // (count - fracBits) is amount of bits shifted relative to fixed point decimal location
    exp = (int32_t(count) - int32_t(fracBits)) + int32_t(fp16ExpBias);

    // Calculate the fp16 fractional part (remaining bits after the leading 1-bit)
    uint32_t xAbsFrac;
    if (count == 0) // Input is zero or denorm
    {
        // Shift frac bits of fixed point input to fill upper bits of fp16 frac
        frac = xAbs << (fp16FracBits - count - 1);
    } else {
        xAbsFrac = xAbs ^ (0x01 << count); // Fractional part excluding leading 1-bit
        frac = RoundFp16(xAbsFrac, count);
    }

    result = (int32_t)PACK_F16(sign, exp, frac);

    return result;
}

unsigned int f32_to_b16_conv(unsigned int x, unsigned int rnd_mode, unsigned int *exceptions) {
    unsigned int result;
    unsigned int sign;
    int exp;
    unsigned int frac;  //, res_frac;

    frac = EXTRACT_F32_FRAC(x);
    exp = EXTRACT_F32_EXP(x);
    sign = EXTRACT_F32_SIGN(x);

    if (exp == 0xFF) {
        // it's either a NaN or infinite
        if (frac != 0) {
            // NaN
            if (((frac >> 22) & 0x1) == 0x0) {
                // signalling NaN
                *exceptions |= F32_EX_INVALID;
            }
            result = 0x7FC0;
        } else {
            // infinity
            result = PACK_B16(sign, 0xFF, 0);
        }
    } else if (exp == 0x0) {
        if (frac != 0) {
            // Denormal
            // Flush to zero
            *exceptions |= (F32_EX_INEXACT | F32_EX_UNDERFLOW);
            result = PACK_B16(sign, 0, 0);
        } else {
            // Zero
            result = PACK_B16(sign, 0, 0);
        }
    } else {
        // Extract lsb, round and sticky bits
        int lsb = frac & 0x10000;
        int round = frac & 0x8000;
        int sticky = ((frac & 0x7fff) != 0) ? 1 : 0;

        // Truncate significand
        frac = frac >> 16;

        // Increment if necessary
        switch (rnd_mode) {
        case F32_RND_NEAREST_EVEN:
            if ((round && lsb) || (round && sticky))
                frac = frac + 1;
            break;
        case F32_RND_TO_ZERO:
            break;
        case F32_RND_PLUS_INF:
            if ((sign == 0) && (round || sticky))
                frac = frac + 1;
            break;
        case F32_RND_MINUS_INF:
            if ((sign == 1) && (round || sticky))
                frac = frac + 1;
            break;
        }

        // Inexact if either round or sticky bit set
        if (round || sticky)
            *exceptions |= F32_EX_INEXACT;

        // Check if rounding caused significand overflow
        if ((frac & 0x80) != 0) {
            frac = 0;
            exp = exp + 1;
        }

        result = PACK_B16(sign, exp, frac);
    }

    return result;
}

void setupInt(host_parsing::DType in_type, host_parsing::DType out_type, host_parsing::DPUInvariantRegisters &regs,
               ActivationFunctionDesc &actFuncDesc, uint8_t out_zero_point) {
    if (in_type == host_parsing::DType::I8 || in_type == host_parsing::DType::U8 || in_type == host_parsing::DType::I4) {
        // I8/U8/I4 in, INT32 convolution, I8/U8/I4 out
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000; // INT32 convolution -> bypass FP clamp/gain
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
        regs.ppe_fp_prelu = 0;

        if (actFuncDesc.funcType == ActivationFunction::leaky_relu) {
            // in this case, we have to convert a high-precision floating-point
            // LeakyReLU alpha value to integer multiply and shift register values
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = actFuncDesc.alphaMult;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = actFuncDesc.alphaShift;
        } else if (actFuncDesc.funcType == ActivationFunction::relu_x) {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
        } else if (actFuncDesc.funcType == ActivationFunction::relu) {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
        } else {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;  // no activation function
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0; // no activation function
        }
    } else {
        // FP16/BF16/FP8 in, FP32 convolution, U8 out
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x004; // FP32 convolution -> INT32 (and eventually U8) out

        // Derive fp _prelu
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
        regs.ppe_bias = 0;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0; // can be overridden by LeakyReLU case

        // FP32 prelu
        if ((actFuncDesc.funcType == ActivationFunction::leaky_relu) || (actFuncDesc.funcType == ActivationFunction::relu) ||
            (actFuncDesc.funcType == ActivationFunction::relu_x)) {
            // for LeakyReLU, apply alpha; for ReLU and ReLUX, apply a negative-X slope of 0
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
            regs.ppe_fp_prelu = actFuncDesc.alphaFP32.u32;
        }
    }

    //
    // U8 offset is added before clamping in VPU2.6
    switch (out_type) {
        case host_parsing::DType::I4:
            regs.ppe_scale_lclamp = -8;
            break;
        case host_parsing::DType::I8:
            regs.ppe_scale_lclamp = -128;
            break;
        case host_parsing::DType::U8:
            regs.ppe_scale_lclamp = 0;
            break;
        case host_parsing::DType::I32:
            regs.ppe_scale_lclamp = 0x80000000;
            break;
        default:
            VPUX_THROW("Unexpected dtype: {0}", static_cast<uint8_t>(out_type));
    }

    if (actFuncDesc.funcType == ActivationFunction::relu_x) {
        regs.ppe_scale_hclamp = (uint32_t)actFuncDesc.clampHigh;
    } else {
        switch (out_type) {
            case host_parsing::DType::I4:
                regs.ppe_scale_hclamp = 7;
                break;
            case host_parsing::DType::I8:
                regs.ppe_scale_hclamp = 127;
                break;
            case host_parsing::DType::U8:
                regs.ppe_scale_hclamp = 255;
                break;
            case host_parsing::DType::I32:
                regs.ppe_scale_hclamp = 0x7FFFFFFF;
                break;
            default:
                VPUX_THROW("Unexpected dtype: {0}", static_cast<uint8_t>(out_type));
        }
    }

    // U8 Quantization logic requires a final addition of the zero point
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = out_zero_point;
}

// FP8/FP16/FP32 out
void setupFloat(host_parsing::DType in_type, host_parsing::DType out_type, host_parsing::DPUInvariantRegisters &regs,
                ActivationFunctionDesc &actFuncDesc) {
    switch (in_type) {
        case host_parsing::DType::I8:
        case host_parsing::DType::U8: {
            VPUX_THROW_UNLESS(out_type == host_parsing::DType::FP32,
                "Input datatype {0} with FP32 output is not supported", static_cast<uint8_t>(in_type));
            // U8 in, INT32 convolution, FP16 out
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000; // INT32 convolution -> bypass FP clamp/gain
                                                                  //
            regs.ppe_misc.ppe_misc_bf.ppe_i32_convert =
                (out_type == host_parsing::DType::FP8) ? 0x2 : 0x1; // INT32 s17.15 fixed-point convert to FP8/FP16

            if (actFuncDesc.funcType == ActivationFunction::leaky_relu) {
                // in this case, we have to convert a high-precision floating-point
                // LeakyReLU alpha value to integer multiply and shift register values
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = actFuncDesc.alphaMult;
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = actFuncDesc.alphaShift;
            } else if (actFuncDesc.funcType == ActivationFunction::relu_x) {
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
            } else if (actFuncDesc.funcType == ActivationFunction::relu) {
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
            } else {
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;  // no activation function
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0; // no activation function
            }

            break;
        }
        case host_parsing::DType::BFP16:
        case host_parsing::DType::FP16:
        case host_parsing::DType::FP8: {
            // FP16 in, FP32 convolution, FP16 out
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
            if (out_type != host_parsing::DType::FP32)
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                    (out_type == host_parsing::DType::FP8) ? 0x003 : 0x001; // FP32 convolution -> FP8/FP16 out

            // FP32 Prelu
            if ((actFuncDesc.funcType == ActivationFunction::leaky_relu) || (actFuncDesc.funcType == ActivationFunction::relu) ||
                (actFuncDesc.funcType == ActivationFunction::relu_x)) {
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
                regs.ppe_fp_prelu =
                    actFuncDesc.alphaFP32.u32; // deliberately apply gain of zero to values less than zero
            } else {
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
            }

            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

            // Do not apply the scaling table to the integer PPE
            regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
            regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
            regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;

            break;
        }
        default: {
            VPUX_THROW("Support for input datatype {0} with FP output is not yet implemented",
                static_cast<uint8_t>(in_type));
        }
    }

    // ReLUX is ReLU with an upper clamp
    if (actFuncDesc.funcType == ActivationFunction::relu_x) {
        uint32_t hclampAsFP16 = static_cast<uint32_t>(fixedPointToFp16((uint32_t)actFuncDesc.clampHigh, 32, 0));
        uint32_t hclampAsFP8 = ((hclampAsFP16 & 0x0000FF00) >> 8);

        // BF16 not yet supported here
        regs.ppe_scale_hclamp = (out_type == host_parsing::DType::FP8) ? hclampAsFP8 : hclampAsFP16;
        regs.ppe_scale_lclamp = 0x80000000;
    } else {
        // ReLU, LeakyReLU, unsupported
        regs.ppe_scale_hclamp = 0x7fffffff;
        regs.ppe_scale_lclamp = 0x80000000;
    }
}

void setupBFloat(host_parsing::DType in_dtype, host_parsing::DType out_type, host_parsing::DPUInvariantRegisters &regs,
                 const ActivationFunctionDesc &actFunc) {
    VPUX_UNUSED(out_type);

    VPUX_THROW_UNLESS(in_dtype != host_parsing::DType::I8 && in_dtype != host_parsing::DType::U8, "X8 in, I32 convolution, BF16 out is not supported by the hardware");

    // FP8/FP16/BF16 in, FP32 convolution, BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x002; // FP32 convolution -> BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 1;     // Round to Nearest, Ties to Even (RNE)

    // FP32 Prelu
    if ((actFunc.funcType == ActivationFunction::leaky_relu) || (actFunc.funcType == ActivationFunction::relu) || (actFunc.funcType == ActivationFunction::relu_x)) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
        regs.ppe_fp_prelu = actFunc.alphaFP32.u32; // deliberately apply gain of zero to values less than zero
    } else {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    }

    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

    // Do not apply the scaling table to the integer PPE
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;

    // ReLUX is ReLU with an upper clamp
    if (actFunc.funcType == ActivationFunction::relu_x) {
        unsigned int exceptions = 0;
        u32f32 hClamp;
        hClamp.f32 = (float)((uint32_t)actFunc.clampHigh);

        regs.ppe_scale_hclamp = f32_to_b16_conv(hClamp.u32, F32_RND_NEAREST_EVEN, &exceptions);
        regs.ppe_scale_lclamp = 0x80000000;
    } else {
        // ReLU, LeakyReLU, unsupported
        regs.ppe_scale_hclamp = 0x7fffffff;
        regs.ppe_scale_lclamp = 0x80000000;
    }
}

VPUIP::MPEMode getMPEFrequentModeFromDPUTasks(mlir::Region& dpuTaskOps) {
    std::unordered_map<VPUIP::MPEMode, size_t> umap;
    for (auto dpuTaskOp : dpuTaskOps.getOps<VPUIP::DPUTaskOp>()) {
        const auto mpeMode = dpuTaskOp.mpe_mode();
        return mpeMode;
        // vpux::printTo(std::cerr, "Successfully extracted {0} mpe mode\n", mpeMode);
        // ++umap[mpeMode];
        // if (umap.size() > 1) {
            // VPUX_THROW("Non-uniform DPU task MPE modes is not supported yet.");
        // }
    }
    return umap.begin()->first;
}

// VPUIP::MPEMode getMPEFrequentModeFromDPUTasks(mlir::Region& dpuTaskOps) {
//     std::unordered_set<vpux::VPUIP::MPEMode> modes;
//     for (auto task : dpuTaskOps.getOps<vpux::VPUIP::DPUTaskOp>()) {
//         modes.insert(task.mpe_mode());
//     }
//     VPUX_THROW_UNLESS(modes.size() == 1, "Encountered nce task with dpu tasks region with inconsistent MPE modes");
//     return *modes.begin();
// }

uint32_t calc_se_size(uint32_t x) {
    uint32_t sz = 0;
    while (x) {
        x >>= 1;
        ++sz;
    }
    // OPEN: should it be treated as a error
    // uint32_t x_orig = x;
    // if (x_orig != (1u << (sz - 1))) {
    // nnLog(MVLOG_WARN, "storage_element_size is %d which is not a power of 2", x_orig);
    // }
    // HW register NCE_DPU_Z_CONFIG.se_z_split has values: 1=16, 2=32....9=4096, 0=8192
    if (sz > 4) {
        sz -= 4;
    } else {
        sz = 1;
    }
    // if Z size is 8192 or bigger, adjust to HW value for 8192
    if (sz >= 10) {
        sz = 0;
        // OPEN: should it be treated as a error
        // nnLog(MVLOG_WARN, "storage_element_size bigger then 8192, HW value adjusted for 8192");
    }
    return sz;
}

// Turn the integer mult and shift into a float (mult / 2**shift)
float integerPreluAlpha(uint32_t activationFunctionMult, uint32_t activationFunctionShift) {
    if (activationFunctionShift > 0)
        return (((float)activationFunctionMult) * (1.0 / pow(2, ((float)activationFunctionShift))));
    else
        return -1.0;
}

// Calculate the % difference between the calculated (HW/integer) alpha and our goal
float integerPreluAlphaDeltaPct(float targetPreluAlpha, float actualPreluAlpha) {
    if (abs(targetPreluAlpha) > 0)
        return abs(targetPreluAlpha - actualPreluAlpha) / abs(targetPreluAlpha);
    else
        return -1.0;
}

// Return -1 if actualAlpha < targetAlpha, 1 otherwise to determine approximation direction
int integerPreluAlphaDeltaSgn(float targetPreluAlpha, float actualPreluAlpha) {
    return (targetPreluAlpha >= actualPreluAlpha) ? 1 : -1;
}

// Approximate the HW integer prelu alpha settings given the target float alpha value in the blob
// Start at the largest values possible in the PPE registers and work backward until the target is reached
// If both fields reach 0 then we can't approximate this alpha value and return failure
bool approximatePreluAlpha(float targetAlpha, ActivationFunctionDesc &actFunctionDesc) {
    // Size of fields in PPE prelu register
    constexpr uint32_t intPreluMultBits = 11;
    constexpr uint32_t intPreluShiftBits = 5;

    int32_t mult = (1 << intPreluMultBits) - 1;
    int32_t shft = (1 << intPreluShiftBits) - 1;
    float approxAlpha = integerPreluAlpha(actFunctionDesc.alphaMult, actFunctionDesc.alphaShift);
    float alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
    int alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);
    float alphaErrorPctPrev = alphaErrorPct;
    int alphaErrorSgnPrev = alphaErrorSgn;

    bool multDescentDone = false;
    bool shftDescentDone = false;
    bool multDescentSuccess = false;
    bool shftDescentSuccess = false;

    // Decrease shift until the sign of the error changes
    while (!shftDescentDone && (shft > 0)) {
        shft--;
        approxAlpha = integerPreluAlpha(mult, shft);
        alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
        alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);

        // Error sign changed, we are as close as we can get with shift
        if (alphaErrorSgnPrev ^ alphaErrorSgn) {
            shftDescentSuccess = true;
            shftDescentDone = true;

            // Adjust for which approximation was closest to the actual alpha
            if (alphaErrorPctPrev < alphaErrorPct)
                shft++;

            approxAlpha = integerPreluAlpha(mult, shft);
        } else {
            alphaErrorPctPrev = alphaErrorPct;
            alphaErrorSgnPrev = alphaErrorSgn;
            if (shft == 0)
                shftDescentDone = true;
        }
    }

    // Decrease mult until the sign of the error changes
    if (shftDescentDone && shftDescentSuccess) {
        approxAlpha = integerPreluAlpha(mult, shft);
        alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
        alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);
        alphaErrorPctPrev = alphaErrorPct;
        alphaErrorSgnPrev = alphaErrorSgn;

        while (!multDescentDone && (mult > 0)) {
            mult--;
            approxAlpha = integerPreluAlpha(mult, shft);
            alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
            alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);

            // Error sign changed, we are as close as we can get with mult
            if (alphaErrorSgnPrev ^ alphaErrorSgn) {
                multDescentSuccess = true;
                multDescentDone = true;

                // Adjust for which approximation was closest to the actual alpha
                if (alphaErrorPctPrev < alphaErrorPct)
                    mult++;

                approxAlpha = integerPreluAlpha(mult, shft);
            } else {
                alphaErrorPctPrev = alphaErrorPct;
                alphaErrorSgnPrev = alphaErrorSgn;
                multDescentDone = (mult == 0);
            }
        }

        // Found a solution
        if (multDescentDone && multDescentSuccess) {
            approxAlpha = integerPreluAlpha(mult, shft);
            actFunctionDesc.alphaMult = mult;
            actFunctionDesc.alphaShift = shft;

            return true;
        }
    }

    // Either mult or shift were 0 before approximation were complete
    VPUX_THROW("Failed to approximate pReLU target alpha: {0} mult: 0x{1} shft: 0x{2} flags: {3}, {4}, {5}, {6}");
}

unsigned int configWorkloadSize(vpux::VPUIP::NCETaskType taskType, unsigned int inputShapeC, unsigned int size) {
    switch (taskType) {
        case vpux::VPUIP::NCETaskType::CONV:
        case vpux::VPUIP::NCETaskType::ELTWISE:
        case vpux::VPUIP::NCETaskType::CMCONV:
            // TODO: There seems to be some kind of untold convention with
            // the compiler that this value will be overwritten in runtime
            // OPEN: check this moment
            if (size != inputShapeC)
                size = inputShapeC;
            break;

        default:
            break;
    }

    return size;
}

uint8_t getODUDTypeSizeBits(host_parsing::OutputTensorDType type)
{
    switch (type)
    {
        //case host_parsing::OutputTensorDType::FP16:
        case host_parsing::OutputTensorDType::BF16:
            return 16;
        case host_parsing::OutputTensorDType::U8F:
        case host_parsing::OutputTensorDType::G8:
        case host_parsing::OutputTensorDType::I8:
            return 8;
        //case host_parsing::OutputTensorDType::FP32:
        case host_parsing::OutputTensorDType::I32:
            return 32;
        case host_parsing::OutputTensorDType::I4:
            return 4;
        case host_parsing::OutputTensorDType::I2:
            return 2;
        case host_parsing::OutputTensorDType::LOG:
            return 4;
        case host_parsing::OutputTensorDType::BIN:
            return 1;
        default:
            return 1;
    }
}

// Round up val by N
template <size_t N>
uint32_t round_up(uint32_t t) {
    return static_cast<uint32_t>((t + N - 1) & ~(N - 1));
}

llvm::SmallVector<std::pair<uint32_t, int32_t>> reduce_dims_for_dma(mlir::MemRefType memref) {
    auto const logicalShape = vpux::getShape(memref);
    auto const logicalStrides = vpux::getStrides(memref);
    auto const order = vpux::DimsOrder::fromType(memref);
    auto const memShape = order.toMemoryOrder(logicalShape);
    auto const memStrides = order.toMemoryOrder(logicalStrides);

    llvm::outs() << " iShapeLogical \n";
    logicalShape.printFormat(llvm::outs());
    llvm::outs() << " inputStridesLogical\n";
    logicalStrides.printFormat(llvm::outs());
    llvm::outs() << " inputOrder\n";
    order.printFormat(llvm::outs());
    llvm::outs() << " inputShapeMemory\n";
    memShape.printFormat(llvm::outs());
    llvm::outs() << " inputStridesMemory\n";
    memStrides.printFormat(llvm::outs());

    auto inner_most_index = memShape.size() - 1;
    llvm::SmallVector<std::pair<uint32_t, int32_t>> finalDims;

    uint32_t previous_size = memShape[MemDim(inner_most_index)];
    int32_t previous_stride_bits = vpux::Bit(memStrides[MemDim(inner_most_index)]).count();

    if (previous_size * memref.getElementTypeBitWidth() < previous_stride_bits) {
        int32_t final_stride = previous_stride_bits / CHAR_BIT;
        uint32_t final_size = previous_size * memref.getElementTypeBitWidth() / CHAR_BIT;

        finalDims.push_back({final_size, final_stride});
    }

    // TODO:: could there be some way to iterate over all MemDim's of a particular shape/order?
    for (int dim = inner_most_index - 1, level = 0; dim > 0; --dim) {
        auto memDim = MemDim(dim);

        uint32_t current_size = memShape[memDim];
        int32_t current_stride_bits = vpux::Bit(memStrides[memDim]).count();

        if (previous_size * previous_stride_bits < current_stride_bits) {
            int32_t final_stride = current_stride_bits / CHAR_BIT;
            uint32_t final_size = (previous_size * previous_stride_bits) / CHAR_BIT;

            finalDims.push_back({final_size, final_stride});
        }

        previous_size = current_size;
        previous_stride_bits = current_stride_bits;
    }

    if (finalDims.size() == 0) {
        uint32_t final_size = (previous_size * previous_stride_bits) / CHAR_BIT;
        int32_t final_stride = final_size;
        finalDims.push_back({final_size, final_stride});
    }

    return finalDims;
}

std::vector<char> exportToBlobELF(mlir::ModuleOp module, mlir::TimingScope& rootTiming, Logger log) {
    std::cerr << "ELF" << '\n';
    log.setName("VPUIP::BackEnd (ELF)");

    log.trace("Extract 'IE.{0}' from Module (ELF)", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Extract 'VPUIP.{0}' from Module (ELF)", VPUIP::GraphOp::getOperationName());
    auto graphOp = VPUIP::GraphOp::getFromModule(module);

    std::deque<host_parsing::BarrierWrapper> barriersList;
    mlir::DenseMap<mlir::Value, std::deque<host_parsing::BarrierWrapper>::iterator> barriers;
    netFunc.walk([&barriersList, &barriers](VPUIP::ConfigureBarrierOp barrierOp) {
        host_parsing::BarrierWrapper wrapper{};
        wrapper.real_id = checked_cast<uint8_t>(barrierOp.id());
        VPUX_THROW_UNLESS(
                barriers.try_emplace(barrierOp.barrier(), barriersList.insert(barriersList.end(), std::move(wrapper)))
                        .second,
                "Encountered the same barrierOp {0} in function twice", barrierOp);
    });

    VPUX_THROW_UNLESS(barriersList.size() <= 32, "Networks with barrriers count more than 32 are not supported");

    for (auto barriersPosition = barriersList.begin(); barriersPosition != barriersList.end(); ++barriersPosition) {
        const auto nextSameId =
                std::find_if(std::next(barriersPosition), barriersList.end(), [&barriersPosition](const auto& barrier) {
                    return barriersPosition->real_id == barrier.real_id;
                });
        barriersPosition->next_same_id = nextSameId != barriersList.end() ? std::distance(nextSameId, barriersList.end()) : -1;
    }

    host_parsing::HostParsedInference hostParsedInference{};
    hostParsedInference.magic = 'E' << 16 | 'L' << 8 | 'F';
    auto& resourceRequirements = hostParsedInference.resource_requirements;

    {
        auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
        resources.walk([&](IERT::ExecutorResourceOp res) {
            const auto kind = res.kind().dyn_cast_or_null<VPUIP::PhysicalProcessorAttr>();
            if (!kind) {
                return;
            }

            if (kind.getValue() != VPUIP::PhysicalProcessor::NCE_Cluster) {
                return;
            }

            VPUX_THROW_UNLESS(resourceRequirements.slice_count == 0, "Encountered more than one resource of kind {0}",
                              kind);
            resourceRequirements.slice_count = checked_cast<uint8_t>(res.count());
        });
    }

    mlir::DenseMap<mlir::Value, std::size_t> constantIndexes;
    std::vector<std::uint8_t> constants;
    netFunc.walk([&constantIndexes, &constants](vpux::Const::DeclareOp constant) {
        auto const insertion = constantIndexes.insert({constant.output(), constants.size()});
        if (insertion.second) {
            const auto content = constant.contentAttr().fold();
            const auto oldSize = constants.size();
            const auto newSize = static_cast<vpux::Byte>(content.getTotalSize()).count();
            constants.resize(oldSize + newSize);
            content.copyTo(llvm::MutableArrayRef<char>(reinterpret_cast<char*>(constants.data()) + oldSize, checked_cast<size_t>(newSize)));
        }
    });

    std::vector<vpux::VPUIP::DmaTask> dmaTasks;
    std::vector<vpux::VPUIP::DmaTask> leadingDmaTasks;
    netFunc.walk([&](vpux::VPUIP::NNDMAOp dmaTask) {
        auto const input = dmaTask.input();
        auto const output = dmaTask.output_buff();

        VPUX_THROW_UNLESS(input, "Encountered DMA operation {0} without input", dmaTask);
        auto const inputType = input.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(inputType, "Encountered DMA operation {0} from which it is impossible to get MemRefType");
        VPUX_THROW_UNLESS(output, "Encountered DMA operation {0} without output", dmaTask);
        auto const outputType = output.getType().dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(outputType, "Encountered DMA operation {0} from which it is impossible to get MemRefType");

        vpux::VPUIP::DmaTask task{};
        auto& dmaDescriptor = task.dmaDescriptor.transaction;
        dmaDescriptor.cfg_link.cfg_bits.type = 1;
        dmaDescriptor.cfg_link.cfg_bits.burst_length = 16;
        dmaDescriptor.cfg_link.cfg_bits.critical = 1;
        dmaDescriptor.cfg_link.cfg_bits.barrier_en = 1;
        dmaDescriptor.length = inputType.getNumElements() * vpux::Byte(vpux::getElemTypeSize(inputType)).count();

        // TODO::can't we have this reduction at a pass at memref level?
        // TODO::need to place some conditions on the DMA, and in some scenarios, may have to do 1*DMA -> n*DMA
        // transaction rewrites
        auto reduced_dims_input = reduce_dims_for_dma(inputType);
        auto reduced_dims_output = reduce_dims_for_dma(outputType);

        if (reduced_dims_input.size() > 2 || reduced_dims_output.size() > 2) {
            VPUX_THROW("cannot reduce dims to 2 for DMA");
        }

        dmaDescriptor.attr2d.src_width = reduced_dims_input[0].first;
        dmaDescriptor.attr2d.src_stride = reduced_dims_input[0].second;
        dmaDescriptor.attr2d.dst_width = reduced_dims_output[0].first;
        dmaDescriptor.attr2d.dst_stride = reduced_dims_output[0].second;

        if (reduced_dims_input.size() == 2 && reduced_dims_output.size() == 2) {
            VPUX_THROW_UNLESS(reduced_dims_input[1].first == reduced_dims_output[1].first,
                              "Dma's don't have equal plane stride");
            dmaDescriptor.src_plane_stride = reduced_dims_input[1].second;
            dmaDescriptor.dst_plane_stride = reduced_dims_output[1].second;

            uint32_t nPlanes = dmaDescriptor.length / reduced_dims_input[1].first;
            VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits");
            dmaDescriptor.num_planes = nPlanes;
        } else if (reduced_dims_input.size() == 2) {
            dmaDescriptor.src_plane_stride = reduced_dims_input[1].second;
            dmaDescriptor.dst_plane_stride = dmaDescriptor.attr2d.dst_stride;

            uint32_t nPlanes = dmaDescriptor.length / reduced_dims_input[1].first;
            VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits");
            dmaDescriptor.num_planes = nPlanes;
        } else if (reduced_dims_output.size() == 2) {
            dmaDescriptor.src_plane_stride = dmaDescriptor.attr2d.src_stride;
            dmaDescriptor.dst_plane_stride = reduced_dims_output[1].second;
            uint32_t nPlanes = dmaDescriptor.length / reduced_dims_output[1].first;
            VPUX_THROW_UNLESS(nPlanes < 256, "nPlanes is only on 8 bits");

            dmaDescriptor.num_planes = nPlanes;
        } else {
            dmaDescriptor.src_plane_stride = dmaDescriptor.attr2d.src_stride;
            dmaDescriptor.dst_plane_stride = dmaDescriptor.attr2d.dst_stride;
            dmaDescriptor.num_planes = 0;
        }

        auto* insertPosition = &dmaTasks;
        if (dmaTask.waitBarriers().empty()) {
            insertPosition = &leadingDmaTasks;
        }

        for (const auto& barrierValue : dmaTask.waitBarriers()) {
            auto barrierOp = barrierValue.getDefiningOp<VPUIP::ConfigureBarrierOp>();
            VPUX_THROW_UNLESS(barrierOp, "Encountered unexpected barrier value {0} in waitBarriers range of {1}",
                            barrierValue, dmaTask);

            const auto& barrier = barrierOp.barrier();
            VPUX_THROW_UNLESS(barriers.count(barrier),
                            "Encountered unpexpected {0}, all bariers must have been parsed already", barrier);
            auto& barrierWrapper = barriers[barrier];

            barrierWrapper->consumer_count++;
            dmaDescriptor.barriers.cons_mask |= 1 << barrierWrapper->real_id;
        }

        for (const auto& barrierValue : dmaTask.updateBarriers()) {
            auto barrierOp = barrierValue.getDefiningOp<VPUIP::ConfigureBarrierOp>();
            VPUX_THROW_UNLESS(barrierOp, "Encountered unexpected barrier value {0} in updateBarriers range of {1}",
                            barrierValue, dmaTask);

            const auto& barrier = barrierOp.barrier();
            VPUX_THROW_UNLESS(barriers.count(barrier),
                            "Encountered unpexpected {0}, all bariers must have been parsed already", barrier);
            auto& barrierWrapper = barriers[barrier];

            barrierWrapper->producer_count++;
            dmaDescriptor.barriers.prod_mask |= 1 << barrierWrapper->real_id;
        }

        auto& inputPatchingInfo = task.input;
        if (auto inputDeclareOp = input.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            inputPatchingInfo.dataOffset = inputDeclareOp.dataIndex();
            inputPatchingInfo.location.memLocation = inputDeclareOp.locale();
            inputPatchingInfo.location.locationIndex = 0;
        } else if (auto inputConstDeclareOp = input.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(constantIndexes.count(input), "Encountered unexpected value {0}", inputConstDeclareOp);
            inputPatchingInfo.dataOffset = constantIndexes[input];
            inputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::GraphFile;
            inputPatchingInfo.location.locationIndex = 0;
        } else if (input.isa<mlir::BlockArgument>()) {
            inputPatchingInfo.dataOffset = 0;  // dataOffset inside input
            inputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
            inputPatchingInfo.location.locationIndex = 0;  // input's index
        } else {
            VPUX_THROW("Encountered unsupported defining op {0} for input of {1}", input, dmaTask);
        }

        auto& outputPatchingInfo = task.output;
        if (auto outputDeclareOp = output.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            outputPatchingInfo.dataOffset = outputDeclareOp.dataIndex();
            outputPatchingInfo.location.memLocation = outputDeclareOp.locale();
            outputPatchingInfo.location.locationIndex = 0;
        } else if (output.isa<mlir::BlockArgument>()) {
            outputPatchingInfo.dataOffset = 0;  // dataOffset inside output
            outputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableOutput;
            outputPatchingInfo.location.locationIndex = 0;  // output's index
        } else {
            VPUX_THROW("Encountered unsupported defining op for output of {0}", dmaTask);
        }

        insertPosition->push_back(std::move(task));
    });

    std::vector<vpux::VPUIP::DPUTask> dpuTasks;
    netFunc.walk([&](vpux::VPUIP::NCEClusterTaskOp nceTask) {
        const auto& input = nceTask.input();
        VPUX_THROW_UNLESS(input, "Encountered DPU operation {0} without input", nceTask);
        const auto& inputShape = vpux::getShape(input);

        // OPEN: how to fill invariant wrapper virual_dep_ since there is no Virtual Dependancy Tracker in compiler
        // invWrapper.invariant_.barriers_.virtual_dep_ = vdt_.add(ti.base_task());

        host_parsing::DPUInvariantWrapper invariant{};
        auto& registers = invariant.invariant.registers;

        const auto inputShapeC = inputShape[vpux::Dims4D::Act::C];
        const auto inputShapeH = inputShape[vpux::Dims4D::Act::H];
        const auto inputShapeW = inputShape[vpux::Dims4D::Act::W];

        // Looks like it's important to fill the data in this particular order, since some register fields are
        // bitfields and memory for next fields could be overwritten during setting a value for current one.
        registers.tensor_size0.tensor_size0_bf.tensor_size_x = checked_cast<uint32_t>(inputShapeW);

        // Assume parent input tensor is the same as input tensor
        registers.tensor_size0.tensor_size0_bf.tensor_size_y = checked_cast<uint32_t>(inputShapeH);
        registers.tensor_size1.tensor_size1_bf.tensor_size_z = checked_cast<uint32_t>(inputShapeC);

        // NOT USED BY RTL. USED BY MODEL TO SUPPORT LEGACY BEHAVIOUR
        registers.z_config.z_config_bf.addr_format_sel = 1;

        const auto inputElementType = input.getType().cast<mlir::ShapedType>().getElementType();

        registers.tensor_mode.tensor_mode_bf.amode = static_cast<uint8_t>(Type2InputDType(inputElementType));

        // OPEN: original version just checked if input data type is not F16
        uint8_t inputZeroPoint = 0;
        if (const auto quantizedType = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
            inputZeroPoint = checked_cast<uint8_t>(quantizedType.getZeroPoint());
            registers.mpe_cfg.mpe_cfg_bf.mpe_actbias = inputZeroPoint;
            registers.tensor_mode.tensor_mode_bf.pad_value = inputZeroPoint;
        } else if (const auto quantizedType = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto zeroPoints = quantizedType.getZeroPoints();
            VPUX_THROW_UNLESS(!zeroPoints.empty(), "Expected to see non-empty zero points array in {0}",
                              inputElementType);

            // OPEN: why do we use only the first element?
            inputZeroPoint = checked_cast<uint8_t>(zeroPoints.front());
            registers.mpe_cfg.mpe_cfg_bf.mpe_actbias = inputZeroPoint;
            registers.tensor_mode.tensor_mode_bf.pad_value = inputZeroPoint;
        }

        auto inputOp = input.getDefiningOp<vpux::VPUIP::DeclareTensorOp>();
        VPUX_THROW_UNLESS(inputOp, "Failed to find defining op for input {0} of {1}", input, nceTask);
        const auto inputSparsityIndex = inputOp.sparsityIndex();

        // OPEN: should it be all bits set and what to do if sparsity index is undefined?
        // Maximum number to be stored in flatbuffers, used for sparsity map table address
        //-if this value is present in the field sparsity_index it means DENSE, otherwise we have SPARSE tensor
        constexpr auto DEFAULT_INDEX = 999999999999999999ULL;  // 60 bits, 18 decimals

        const auto isInputDense = !inputSparsityIndex.hasValue() || inputSparsityIndex.getValue() == DEFAULT_INDEX;
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = isInputDense;

        const auto& weights = nceTask.weights();
        VPUX_THROW_UNLESS(weights, "Encountered DPU operation {0} without weights", nceTask);
        const auto weightsElementType = weights.getType().cast<mlir::ShapedType>().getElementType();

        const auto taskType = nceTask.task_type();

        registers.tensor_mode.tensor_mode_bf.wmode = static_cast<uint8_t>(
            // OPEN: why weights data type is fixed in this case?
            taskType == vpux::VPUIP::NCETaskType::MAXPOOL ? host_parsing::MpeActivationWeightDtype::I8 : Type2WeightsDType(weightsElementType)
        );

        // OPEN: following weights registers were not set in case of MAXPOOL

        // OPEN: original version just checked if weights data type is U8
        uint8_t weightsZeroPoint = 0;
        uint16_t weightsScale = 1;
        uint8_t weightsShift = 0;
        if (const auto quantizedType = weightsElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
            weightsZeroPoint = checked_cast<uint8_t>(quantizedType.getZeroPoint());
            weightsScale = vpux::getQuantMultFromScale(quantizedType.getScale());
            weightsShift = vpux::getQuantShiftFromScale(quantizedType.getScale());
        } else if (const auto quantizedType = weightsElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto zeroPoints = quantizedType.getZeroPoints();
            VPUX_THROW_UNLESS(!zeroPoints.empty(), "Expected to see non-empty zero points array in {0}",
                              weightsElementType);

            // OPEN: why do we use only the first element?
            weightsZeroPoint = zeroPoints.front();
            const auto scales = quantizedType.getScales();
            VPUX_THROW_UNLESS(!scales.empty(), "Expected to see non-empty scales array in {0}", weightsElementType);
            weightsScale = checked_cast<uint16_t>(getQuantMultFromScale(scales.front()));
            weightsShift = checked_cast<uint8_t>(getQuantShiftFromScale(scales.front()));
        }
        registers.mpe_cfg.mpe_cfg_bf.mpe_wtbias = weightsZeroPoint;

        if (taskType == vpux::VPUIP::NCETaskType::AVEPOOL) {
            if (weightsElementType.isUnsignedInteger(CHAR_BIT * sizeof(uint8_t)) ||
                weightsElementType.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
                registers.elops_wload.elops_wload_bf.pool_wt_data = (1 << CHAR_BIT) | 1;
            } else if (weightsElementType.isF16()) {
                registers.elops_wload.elops_wload_bf.pool_wt_data = ngraph::float16(1.0f);
            } else if (weightsElementType.isBF16()) {
                u32f32 weightsData{};
                weightsData.f32 = 1.0f;

                registers.elops_wload.elops_wload_bf.pool_wt_data =
                        f32_to_b16_conv(weightsData.u32, F32_RND_NEAREST_EVEN, 0);
            } else {
                VPUX_THROW("Encountered unsupported weights data type {0} for {1}", weightsElementType, nceTask);
            }
        }

        registers.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt = 1;
        const auto mpeFrequencyMode = getMPEFrequentModeFromDPUTasks(nceTask.variants());
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = static_cast<uint8_t>(
                mpeFrequencyMode == vpux::VPUIP::MPEMode::VECTOR ? host_parsing::MPEGrid::GRID_16x1
                                                                 : host_parsing::MPEGrid::GRID_4x4);

        uint8_t kernelH = 1;
        uint8_t kernelW = 1;
        if (const auto kernelSizeAttr = nceTask.kernel_sizeAttr()) {
            const auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);
            kernelH = checked_cast<uint8_t>(kernelSize[0]);
            kernelW = checked_cast<uint8_t>(kernelSize[1]);
        }

        VPUX_THROW_UNLESS(KERNEL_SIZE_MIN <= kernelH && kernelH < KERNEL_SIZE_MAX,
                          "Encountered nce task {0} with kernel height {1} out of supported range [{2}..{3}]", nceTask,
                          kernelH, KERNEL_SIZE_MIN, KERNEL_SIZE_MAX);

        VPUX_THROW_UNLESS(KERNEL_SIZE_MIN <= kernelW && kernelW < KERNEL_SIZE_MAX,
                          "Encountered nce task {0} with kernel width {1} out of supported range [{2}..{3}]", nceTask,
                          kernelW, KERNEL_SIZE_MIN, KERNEL_SIZE_MAX);

        registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y = kernelH;
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x = kernelW;

        int16_t kernelStrideH = 1;
        int16_t kernelStrideW = 1;
        if (const auto kernelStridesAttr = nceTask.kernel_stridesAttr()) {
            const auto kernelStrides = parseIntArrayAttr<int64_t>(kernelStridesAttr);
            kernelStrideH = checked_cast<int16_t>(kernelStrides[0]);
            kernelStrideW = checked_cast<int16_t>(kernelStrides[1]);
        }

        VPUX_THROW_UNLESS(KERNEL_STRIDE_MIN <= kernelStrideH && kernelStrideH < KERNEL_STRIDE_MAX,
                          "Encountered nce task {0} with kernel stride height {1} out of supported range [{2}..{3}]",
                          nceTask, kernelStrideH, KERNEL_STRIDE_MIN, KERNEL_STRIDE_MAX);

        VPUX_THROW_UNLESS(KERNEL_STRIDE_MIN <= kernelStrideW && kernelStrideW < KERNEL_STRIDE_MAX,
                          "Encountered nce task {0} with kernel stride width {1} out of supported range [{2}..{3}]",
                          nceTask, kernelStrideW, KERNEL_STRIDE_MIN, KERNEL_STRIDE_MAX);

        // OPEN: data types are inconsistent compiler/registers/runtime logic
        int16_t kernelPadL = 0;
        int16_t kernelPadR = 0;
        int16_t kernelPadT = 0;
        int16_t kernelPadB = 0;
        if (const auto kernelPaddingAttr = nceTask.kernel_paddingAttr()) {
            const auto kernelPadding = parseIntArrayAttr<int64_t>(kernelPaddingAttr);
            kernelPadL = checked_cast<int16_t>(kernelPadding[0]);
            kernelPadR = checked_cast<int16_t>(kernelPadding[1]);
            kernelPadT = checked_cast<int16_t>(kernelPadding[2]);
            kernelPadB = checked_cast<int16_t>(kernelPadding[3]);
        }

        registers.tensor_mode.tensor_mode_bf.stride = kernelStrideW - 1;
        if (kernelStrideH != kernelStrideW) {
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y = kernelStrideH - 1;
        }

        // When the activations and weights are of different types,
        // MPE_MODE must be configured to the larger of the 2 data types.
        registers.mpe_cfg.mpe_cfg_bf.mpe_mode =
                std::min(registers.tensor_mode.tensor_mode_bf.amode, registers.tensor_mode.tensor_mode_bf.wmode);

        const auto& output = nceTask.output_buff();
        VPUX_THROW_UNLESS(output, "Encountered DPU operation {0} without output", nceTask);
        const auto& outputShape = vpux::getShape(output);
        // const auto outputShapeC = outputShape[vpux::Dims4D::Act::C];
        auto outputOp = output.getDefiningOp<vpux::VPUIP::DeclareTensorOp>();
        VPUX_THROW_UNLESS(outputOp, "Failed to find defining op for output {0} of {1}", output, nceTask);
        const auto outputSparsityIndex = outputOp.sparsityIndex();

        const auto isOutputDense = !outputSparsityIndex.hasValue() || outputSparsityIndex.getValue() == DEFAULT_INDEX;

        // TODO: This is an estimate based on what's done above for KMB. Nothing in the POC runtime that sets
        // this, so setting to maximum values for now.
        // invariant.odu_be_size = invariant.odu_be_cnt = 2047; // max
        // registers.odu_be_size = registers.odu_be_cnt = 0;

        // ODU SEs size calculated from output z dimension for 2.7
        // registers.se_size = 0;

        const auto outputElementType = output.getType().cast<mlir::ShapedType>().getElementType();
        registers.odu_cfg.odu_cfg_bf.dtype = static_cast<uint8_t>(Type2OutputDType(outputElementType));
        // invariant.odu_cfg.odu_cfg_bf.mode = 0; // FIXME: how to handle if superdense ?

        switch (mpeFrequencyMode) {
        case vpux::VPUIP::MPEMode::VECTOR:
            // OPEN: registers.odu_cfg.odu_cfg_bf.grid is 1 bit, but host_parsing::ODUGrid has 3 values
            registers.odu_cfg.odu_cfg_bf.grid = static_cast<uint8_t>(host_parsing::ODUGrid::GRID_16x1);
            registers.odu_cfg.odu_cfg_bf.nthw = static_cast<uint8_t>(host_parsing::ODUNthw::NTHW_1);
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign =
                    static_cast<uint8_t>(host_parsing::MPEGrid::GRID_16x1);
            break;
        // OPEN: were not present in VPUX before
        case vpux::VPUIP::MPEMode::CUBOID_4x16:
            registers.odu_cfg.odu_cfg_bf.grid = static_cast<uint8_t>(host_parsing::ODUGrid::GRID_4x4);
            registers.odu_cfg.odu_cfg_bf.nthw = static_cast<uint8_t>(host_parsing::ODUNthw::NTHW_4);
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign =
                    static_cast<uint8_t>(host_parsing::MPEGrid::GRID_4x4);
            break;
        case vpux::VPUIP::MPEMode::CUBOID_8x16:
            registers.odu_cfg.odu_cfg_bf.grid = static_cast<uint8_t>(host_parsing::ODUGrid::GRID_4x4);
            registers.odu_cfg.odu_cfg_bf.nthw = static_cast<uint8_t>(host_parsing::ODUNthw::NTHW_8);
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign =
                    static_cast<uint8_t>(host_parsing::MPEGrid::GRID_4x4);
            break;
        case vpux::VPUIP::MPEMode::CUBOID_16x16:
            registers.odu_cfg.odu_cfg_bf.grid = static_cast<uint8_t>(host_parsing::ODUGrid::GRID_4x4);
            registers.odu_cfg.odu_cfg_bf.nthw = static_cast<uint8_t>(host_parsing::ODUNthw::NTHW_16);
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign =
                    static_cast<uint8_t>(host_parsing::MPEGrid::GRID_4x4);
            break;
        default:
            VPUX_THROW("Encountered unknown mpe frequency mode {0}", mpeFrequencyMode);
            break;
        };

        registers.odu_cfg.odu_cfg_bf.write_ac = 1;                // Always write data out!
        registers.odu_cfg.odu_cfg_bf.write_pt = !isOutputDense;   // Enable/Disable output SE table generation
        registers.odu_cfg.odu_cfg_bf.write_sp = !isOutputDense;   // Enable/Disable output sparsity map generation
        registers.odu_cfg.odu_cfg_bf.sp_out_en = !isOutputDense;  // Enable/Disable compression of output activations

        // OPEN: seems swizzling key is not supported by VPUx yet
        // registers.odu_cfg.odu_cfg_bf.swizzle_key = srcInvariant.output_data->swizzling_key;

        // OPEN: VPUx always writes odu_permutation as ZXY
        // registers.odu_cfg.odu_cfg_bf.permutation = 0;

        // OPEN: do similar way as for input
        uint8_t zeroPoint = 0;
        uint16_t scale = 1;
        uint8_t shift = 0;
        if (const auto quantizedType = outputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
            zeroPoint = checked_cast<uint8_t>(quantizedType.getZeroPoint());
            scale = vpux::getQuantMultFromScale(quantizedType.getScale());
            shift = vpux::getQuantShiftFromScale(quantizedType.getScale());
        } else if (const auto quantizedType = outputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto zeroPoints = quantizedType.getZeroPoints();
            VPUX_THROW_UNLESS(!zeroPoints.empty(), "Expected to see non-empty zero points array in {0}",
                              outputElementType);
            // OPEN: why just the first element
            zeroPoint = checked_cast<uint8_t>(zeroPoints.front());

            const auto scales = quantizedType.getScales();
            VPUX_THROW_UNLESS(!scales.empty(), "Expected to see non-empty scales array in {0}", outputElementType);
            scale = checked_cast<uint16_t>(getQuantMultFromScale(scales.front()));
            shift = checked_cast<uint16_t>(getQuantShiftFromScale(scales.front()));
        }

        // OPEN: seems like runtime expects output always to have quant_zero
        registers.odu_cfg.odu_cfg_bf.sp_value = isOutputDense ? 0 : zeroPoint;

        registers.te_dim1.te_dim1_bf.te_dim_x = checked_cast<uint16_t>(outputShape[vpux::Dims4D::Act::W] - 1);
        registers.te_dim0.te_dim0_bf.te_dim_y = checked_cast<uint16_t>(outputShape[vpux::Dims4D::Act::H] - 1);

        // TODO: Why isn't this simply "output->dimensions[Z] - 1" ?
        // TODO: For channel-major output this seems to be incorrect

        {
            const auto outputStrides = vpux::getStrides(output);
            // OPEN: check get correct stride values (it is NHWC!)
            const auto strideW = vpux::Byte(outputStrides[vpux::Dims4D::Act::W]).count();
            const auto strideC = vpux::Byte(outputStrides[vpux::Dims4D::Act::C]).count();
            VPUX_THROW_UNLESS(strideC > 0, "Encountered nce task {0} with zero stride by C", nceTask);
            // OPEN: strideW / strideC or vice versa
            registers.te_dim0.te_dim0_bf.te_dim_z = checked_cast<uint16_t>((strideW / strideC) - 1);
        }

        registers.base_ptr_a = 0x1;
        registers.base_ptr_b = 0x403;
        // OPEN: VPUx assumes parent output is the same as output
        // if (!isOutputDense && outputShape[vpux::Dims4D::Act::H] != parentOutputShape[vpux::Dims4d::Act::H]) {
        // OPEN: looks a bit strange that values are constant
        // registers.base_ptr_a = (0 << 9) | (1 << 0);
        // registers.base_ptr_b = (2 << 9) | (3 << 0);
        // }

        // OPEN: what with other indexes
        // OPEN: VPUx always writes odu offset as 0
        // registers.base_adr[0] = 0;

        // auto weightsOp = weights.getDefiningOp<vpux::Const::DeclareOp>();
        // VPUX_THROW_UNLESS(weightsOp, "Failed to find defining op for weights {0} of {1}", weights, nceTask);
        // const auto weightsSparsityIndex = weightsOp.sparsityIndex();

        const auto isWeightsDense = true;
        // const auto isWeightsDense = !weightsSparsityIndex.hasValue() || weightsSparsityIndex.getValue() ==
        // DEFAULT_INDEX;

        if (taskType == vpux::VPUIP::NCETaskType::CONV) {
            registers.tensor_mode.tensor_mode_bf.zm_input = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

            // OPEN: sparse weights are not supported by VPUx
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = isWeightsDense;

            // OPEN: VPUx assumes parent input and input are the same (so here I do nothing)
            // check SetupInvariant_SOH & SetupInvariant_SOH_Input & Input Size if-block after

            const auto inputStorageElementSize = inputOp.storageElementSize();

            if (!isInputDense && inputStorageElementSize.hasValue() && inputStorageElementSize.getValue() != 0) {
                const auto storageElementSize = checked_cast<uint32_t>(inputStorageElementSize.getValue());
                registers.z_config.z_config_bf.se_z_split = calc_se_size(storageElementSize);
                const auto inputChannelsCount = inputShape[vpux::Dims4D::Act::C];
                registers.z_config.z_config_bf.num_ses_in_z_dir = (inputChannelsCount / storageElementSize) - 1;
                if (inputChannelsCount % storageElementSize) {
                    registers.z_config.z_config_bf.num_ses_in_z_dir++;
                    // OPEN: should it be treated as a error
                    log.warning("input channels count is not divisible by storage element size");
                }
            }
        } else {
            VPUX_THROW("NCE task type other than convolution is not supported yet");
        }

        registers.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
        registers.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;  // Serialised in fixed function
        registers.ppe_fp_prelu = 1;                           // Derive from ppe_prelu_mult & ppe_prelu_shift
        registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1; // Set based on data types, if we see float point - don't bypass!

        const auto outputZeroPoint = taskType == vpux::VPUIP::NCETaskType::MAXPOOL ? 0 : zeroPoint;
        VPUX_THROW_UNLESS(areSupportedInputOutputTypes(Type2DType(inputElementType), Type2DType(outputElementType)),
                          "Encountered unsupported data types configurations: input ({0}) and output ({1})",
                          inputElementType, outputElementType);

        SmallVector<uint8_t> ppeList;
        auto clampLow = std::numeric_limits<int32_t>::min();
        auto clampHigh = std::numeric_limits<int32_t>::max();
        int32_t LreluMult = 1;
        uint32_t LreluShift = 0;
        ::llvm::Optional<SmallVector<uint16_t>> ppeQuantMult;
        ::llvm::Optional<SmallVector<uint8_t>> ppeQuantShift;
        ::llvm::Optional<int8_t> ppeQuantPostShift;

        for (auto ppeOp : nceTask.ppe().getOps<VPUIP::PPETaskOp>()) {
            const auto type = ppeOp.ppe_layer_type();
            if (type != VPUIP::PPELayerType::NOOP) {
                ppeList.push_back(static_cast<uint8_t>(type));
            }
            if (ppeOp.clamp_low().hasValue()) {
                clampLow = checked_cast<int32_t>(ppeOp.clamp_low().getValue());
            }
            if (ppeOp.clamp_high().hasValue()) {
                clampHigh = checked_cast<int32_t>(ppeOp.clamp_high().getValue());
            }
            if (ppeOp.lrelu_mult().hasValue()) {
                LreluMult = checked_cast<int32_t>(ppeOp.lrelu_mult().getValue());
            }
            if (ppeOp.lrelu_shift().hasValue()) {
                LreluShift = checked_cast<uint32_t>(ppeOp.lrelu_shift().getValue());
            }
            if (ppeOp.quant_mult().hasValue()) {
                ppeQuantMult = parseIntArrayAttr<uint16_t>(ppeOp.quant_mult().getValue());
            }
            if (ppeOp.quant_shift().hasValue()) {
                ppeQuantShift = parseIntArrayAttr<uint8_t>(ppeOp.quant_shift().getValue());
            }
            if (ppeOp.quant_post_shift().hasValue()) {
                ppeQuantPostShift = checked_cast<int8_t>(ppeOp.quant_post_shift().getValue());
            }
        }
        VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

        ActivationFunctionDesc actFuncDesc;

        if (!ppeList.empty() && LreluShift != 0) {
            // LeakyReLU: alpha slope derived according to Alessandro's Fathom test script as follows (ca. line 87:)
            // https://github.com/movidius/Fathom/blob/master/scripts/validation_test_script/mix_precision_blobs.py
            // scale_shift_to_fp(scale,shift): scale * 2 ** (-float(shift))
            // scale_shift_to_fp(ppe_ops["Lrelu_Mult"], ppe_ops["Lrelu_Shift"])
            auto lReluMult  = checked_cast<float>(LreluMult);
            auto lReluShift = checked_cast<int8_t>(LreluShift & 0xFF);

            actFuncDesc.funcType = ActivationFunction::leaky_relu;
            actFuncDesc.alpha = std::ldexp(lReluMult, -lReluShift);

            if (inputElementType.isUnsignedInteger(CHAR_BIT * sizeof(uint8_t)) ||
                inputElementType.isSignedInteger  (CHAR_BIT * sizeof( int8_t)) ||
                inputElementType.isSignedInteger  (CHAR_BIT * sizeof(int32_t))) {
                VPUX_THROW_UNLESS(approximatePreluAlpha(actFuncDesc.alpha, actFuncDesc), "Failed to approximate PReLU alpha");
            } else {
                if ((checked_cast<uint32_t>(clampHigh) == 0x7FFFFFFF ||
                     checked_cast<uint32_t>(clampHigh) == 0x00000000) && checked_cast<uint32_t>(clampLow) == 0) {
                    actFuncDesc.funcType = ActivationFunction::relu;
                    actFuncDesc.alpha = -0.0; // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                              // (0x00000000), not negative zero (0x80000000)
                } else if (checked_cast<uint32_t>(clampHigh) < 0x7FFFFFFF && checked_cast<uint32_t>(clampLow) == 0) {
                    actFuncDesc.funcType = ActivationFunction::relu_x;
                    actFuncDesc.alpha = -0.0; // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                              // (0x00000000), not negative zero (0x80000000)
                } else {
                    actFuncDesc.funcType = ActivationFunction::no_activation_function;
                }
            }

            actFuncDesc.alphaFP32.f32 = actFuncDesc.alpha; // alpha (accessible as uint32_t FP32 bit pattern in .u)
            actFuncDesc.clampHigh = clampHigh;
            actFuncDesc.clampLow = clampLow;
        }

        const auto inputDType = Type2DType(inputElementType);
        const auto outputDType = Type2DType(outputElementType);
        const auto weightsDType = Type2DType(weightsElementType);

        switch (outputDType) {
            case host_parsing::DType::I4:
            case host_parsing::DType::I8:
            case host_parsing::DType::U8:
            case host_parsing::DType::I32:
                setupInt(inputDType, outputDType, registers, actFuncDesc, outputZeroPoint);
                break;
            case host_parsing::DType::FP8:
            case host_parsing::DType::FP16:
            case host_parsing::DType::FP32:
                setupFloat(inputDType, outputDType, registers, actFuncDesc);
                break;
            case host_parsing::DType::BFP16:
                setupBFloat(inputDType, outputDType, registers, actFuncDesc);
                break;
            default:
                VPUX_THROW("only U8, I8, FP16 are currently supported for BF16 out");
        }

        const auto isFP16 = inputDType == host_parsing::DType::FP16 || inputDType == host_parsing::DType::BFP16;
        const auto isFP = isFP16 || inputDType == host_parsing::DType::FP32 || inputDType == host_parsing::DType::FP8;

        if (taskType == vpux::VPUIP::NCETaskType::ELTWISE) {
            // Set PPE to read quant values from registers for eltwise since there
            // are no weights tables
            registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

            // OPEN: ppe_task.rounding seems to be unintialized
            // registers.ppe_scale.ppe_scale_bf.ppe_scale_round = to_underlying(srcInvariant.ppe_task.rounding);

            // For supporting the MTL-style scales case, mult/shift will need to be adjusted along with the
            // code in eltwise.cpp
            registers.ppe_scale.ppe_scale_bf.ppe_scale_mult = scale;
            registers.ppe_scale.ppe_scale_bf.ppe_scale_shift = shift;
            registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = inputZeroPoint;
            registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = weightsZeroPoint;

            if (isFP) {
                registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0x1;
                registers.ppe_fp_scale = 0x3f800000; // fp32 equiv of 1
                registers.ppe_fp_bias = 0x0;
            }
        } else if (taskType == vpux::VPUIP::NCETaskType::MAXPOOL) {
            registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
            registers.ppe_scale.ppe_scale_bf.ppe_scale_round = 0x3; // 0x3 - no round
            registers.ppe_scale.ppe_scale_bf.ppe_scale_mult = 0x1;
            registers.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0x0;
            registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0x0;
            registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0x0;

            if (isFP16) {
                registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0x1;
                registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                    0x0; // FP16 MaxPool result is already FP16 with CRL FP MAC => no conversion
            }
            if (isFP) {
                registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0x1;
                registers.ppe_fp_scale = 0x3f800000; // fp32 equiv of 1
                registers.ppe_fp_bias = 0x0;
            }
        } else if (registers.elops_wload.elops_wload_bf.pool_wt_rd_dis && taskType == vpux::VPUIP::NCETaskType::AVEPOOL) {
            registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
            u32f32 fp32_scale;
            fp32_scale.f32 = 1.0f / (kernelH * kernelW);
            switch (weightsDType) {
                case host_parsing::DType::I8:
                case host_parsing::DType::U8:
                    registers.ppe_scale.ppe_scale_bf.ppe_scale_mult = weightsScale;
                    registers.ppe_scale.ppe_scale_bf.ppe_scale_shift = weightsShift;
                    break;
                case host_parsing::DType::FP16:
                case host_parsing::DType::BFP16:
                    registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 1;
                    registers.ppe_fp_scale = fp32_scale.u32;
                    break;
                default:
                    break;
            }
        }

        std::vector<vpux::VPUIP::DPUVariantTask> variantWrappers;
        for (auto dpuTaskOp : nceTask.variants().getOps<vpux::VPUIP::DPUTaskOp>()) {
            // OPEN: using invariant's index here
            host_parsing::DPUVariantWrapper variantWrapper{};
            variantWrapper.invariant_index = checked_cast<uint32_t>(dpuTasks.size());
            // OPEN: there is no task id field in host_parsing::DPUVariant

            auto& registers = variantWrapper.variant.registers;

            const auto start = parseIntArrayAttr<int64_t>(dpuTaskOp.start());
            const auto end = parseIntArrayAttr<int64_t>(dpuTaskOp.end());
            const auto pad = dpuTaskOp.pad();

            auto stride_w = kernelStrideW;
            auto stride_h = kernelStrideH;
            auto K_w = kernelW;
            auto K_h = kernelH;

            auto global_PT = kernelPadT;
            auto global_PL = kernelPadL;

            auto local_PT = checked_cast<int16_t>(pad.top().getInt());
            auto local_PB = checked_cast<int16_t>(pad.bottom().getInt());
            auto local_PL = checked_cast<int16_t>(pad.left().getInt());
            auto local_PR = checked_cast<int16_t>(pad.right().getInt());

            auto output_start_x = checked_cast<int16_t>(start[0]);
            auto output_start_y = checked_cast<int16_t>(start[1]);
            auto output_start_z = checked_cast<int16_t>(start[2]);

            auto output_end_x = checked_cast<int16_t>(end[0]);
            auto output_end_y = checked_cast<int16_t>(end[1]);
            auto output_end_z = checked_cast<int16_t>(end[2]);

            auto op_size_x = output_end_x - output_start_x + 1;
            auto op_size_y = output_end_y - output_start_y + 1;
            auto op_size_z = output_end_z - output_start_z + 1;

            registers.weight_num = op_size_z;

            switch (taskType) {
                case vpux::VPUIP::NCETaskType::CONV:
                    registers.weight_size = inputShapeC * kernelW * kernelH;
                break;
                case vpux::VPUIP::NCETaskType::DWCONV:
                case vpux::VPUIP::NCETaskType::AVEPOOL:
                case vpux::VPUIP::NCETaskType::MAXPOOL:
                    registers.weight_size = op_size_z * kernelW * kernelH;
                break;
                case vpux::VPUIP::NCETaskType::ELTWISE:
                    registers.weight_size = inputShapeW * inputShapeH * inputShapeC;
                break;
                default:
                    VPUX_THROW("Can't setup weight size. Layer type unknown : {0}", taskType);
            }

            // OPEN: why negate, no just dense
            registers.offset_addr.offset_addr_bf.dense_se = !isInputDense;
            registers.offset_addr.offset_addr_bf.conv_cond = nceTask.is_continuedAttr() != nullptr;

            registers.workload_size0.workload_size0_bf.workload_size_x = stride_w * (op_size_x - 1) + K_w - local_PL - local_PR;
            registers.workload_size0.workload_size0_bf.workload_size_y = stride_h * (op_size_y - 1) + K_h - local_PT - local_PB;
            registers.workload_size1.workload_size1_bf.workload_size_z = configWorkloadSize(taskType, inputShapeC, op_size_z);
            registers.workload_size1.workload_size1_bf.pad_count_up = local_PT;
            registers.workload_size1.workload_size1_bf.pad_count_down = local_PB;
            registers.workload_size1.workload_size1_bf.pad_count_left = local_PL;
            registers.workload_size1.workload_size1_bf.pad_count_right = local_PR;
            registers.workload_start0.workload_start0_bf.workload_start_x = (output_start_x * stride_w) - global_PL + local_PL;
            registers.workload_start0.workload_start0_bf.workload_start_y = (output_start_y * stride_h) - global_PT + local_PT;
            registers.workload_start1.workload_start1_bf.workload_start_z = configWorkloadSize(taskType, inputShapeC, op_size_z);
            registers.te_beg1.te_beg1_bf.te_beg_x = output_start_x;
            registers.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
            registers.te_beg0.te_beg0_bf.te_beg_z = output_start_z;

            registers.te_end1.te_end1_bf.te_end_x = output_end_x;
            registers.te_end0.te_end0_bf.te_end_y = output_end_y;
            registers.te_end0.te_end0_bf.te_end_z = output_end_z;

            auto& variant = variantWrapper.variant;
            variant.weight_table_offset = output_start_z;

            if (taskType == vpux::VPUIP::NCETaskType::DWCONV || taskType == vpux::VPUIP::NCETaskType::MAXPOOL) {
                registers.workload_start1.workload_start1_bf.workload_start_z = output_start_z;
                registers.workload_size1.workload_size1_bf.workload_size_z = op_size_z;
            } else if (inputShapeC < 16) {
                registers.workload_start1.workload_start1_bf.workload_start_z = 0;
                registers.workload_size1.workload_size1_bf.workload_size_z = 16;
            } else {
                // All input channels required for one output channel
                registers.workload_start1.workload_start1_bf.workload_start_z = 0;
                registers.workload_size1.workload_size1_bf.workload_size_z = inputShapeC;
            }

            // Split over K, and also streaming over K for now ....
            // ODU has a view of the full output tensor, yet as an optimization
            // in each cluster we bring weights and weight_table portions for each
            // output channel subset we compute in that particular cluster
            // OPEN: assuming output is the same as parent output
            // if (srcInvariant.output_data->dimensions[Z] !=
                // srcInvariant.parent_output_tensor->dimensions[Z]) {
                // if (srcInvariant.output_data->locale_index.size() > 1) {
                    // nnLog(MVLOG_INFO, "Using symmetric SoK: %u instead of %u",
                        // variant.weight_table_offset % srcInvariant.output_data->dimensions[Z],
                        // variant.weight_table_offset);

                    // variant.weight_table_offset %= srcInvariant.output_data->dimensions[Z];
                // } else {
                    // OPEN: check this out
                    // Fathom can split an output among invariants but weights don't need to be adjusted
                    // The blob has the correct offsets already
                    // TODO: What about Fathom blobs which are really SOH/SOK. Is the above logic sufficient?
                    // nnLog(MVLOG_WARN, "Invariant Z dim different than parent but no slices to broadcast");
                // }
            // }

            // Point into the 16 byte weight table entry, corresponding to the output channels subset
            // OPEN: not sure about this as well
            variant.weight_table_offset = variant.weight_table_offset << 4;
            // OPEN: invariant.output_sparsity_offset is new field which looks like here would be always 0
            // OPEN: invariant.odu_offset is always 0 from compiler point of view
            variant.output_sparsity_offset = invariant.invariant.output_sparsity_offset + 0;

            auto bpp = round_up<8>(getODUDTypeSizeBits(Type2OutputDType(outputElementType))) >> 3;

            // SetupVariant SOH
            {
                auto output_start_y = checked_cast<int16_t>(start[1]);
                auto output_end_y = checked_cast<int16_t>(end[1]);

                // OPEN: VPUx always assumes output is the same as parent output
                // if (srcInvariant.output_data->dimensions[Y] != srcInvariant.parent_output_tensor->dimensions[Y]) {
                //     unsigned int lines_per_cluster =
                //         SOH_LinesPerCluster(srcInvariant.parent_output_tensor->dimensions[Y],
                //                             srcInvariant.output_data->dimensions[Y], cluster_count);

                //     output_start_y %= lines_per_cluster;
                //     output_end_y %= lines_per_cluster;
                //     variant.registers.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
                //     variant.registers.te_end0.te_end0_bf.te_end_y = output_end_y;

                //     if (((unsigned)output_start_y > srcInvariant.output_data->dimensions[Y]) ||
                //         ((unsigned)output_end_y > srcInvariant.output_data->dimensions[Y]))
                //         nnLog(MVLOG_WARN, "SOH workload still too big: %u-%u, tensor dim_y %lu", output_start_y, output_end_y,
                //             srcInvariant.output_data->dimensions[Y]);

                //     // Workload start needs adjustment if SOH was not set in invariant
                //     if (invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment == 0) {
                //         auto stride_h = srcInvariant.kernel_strideH;
                //         auto global_PT = srcInvariant.kernel_padTop;
                //         auto local_PT = srcVariant.padTop;
                //         variant.registers.workload_start0.workload_start0_bf.workload_start_y =
                //             (output_start_y * stride_h) - global_PT + local_PT;
                //     }

                //     bool is_out_dense = srcInvariant.output_data->data.sparsity_index == DEFAULT_INDEX;
                //     if (!is_out_dense)
                //         variant.output_sparsity_offset |= srcInvariant.output_data->locale_index[0] << 1;
                // }

                op_size_y = output_end_y - output_start_y + 1;
            }

            invariant.invariant.output_sparsity_offset += op_size_y * op_size_x * op_size_z * bpp;

            // SetupVariant_NTHW_NTK
            {
                // Sets up on NTHW on IDU
                switch (mpeFrequencyMode) {
                    case vpux::VPUIP::MPEMode::VECTOR:
                        registers.offset_addr.offset_addr_bf.nthw_ntk = static_cast<uint8_t>(host_parsing::IDUNthw_Ntk::IDU_8_8);
                        break;
                    case vpux::VPUIP::MPEMode::CUBOID_4x16: // NTH = 1, NTW=4, NTK = 16 (4, 16)
                        registers.offset_addr.offset_addr_bf.nthw_ntk = static_cast<uint8_t>(host_parsing::IDUNthw_Ntk::IDU_4_16);
                        break;
                    case vpux::VPUIP::MPEMode::CUBOID_8x16: // NTH = 2, NTW=4, NTK = 8 (8, 8)
                        registers.offset_addr.offset_addr_bf.nthw_ntk = static_cast<uint8_t>(host_parsing::IDUNthw_Ntk::IDU_8_8);
                        break;
                    case vpux::VPUIP::MPEMode::CUBOID_16x16: // NTH = 4, NTW=4, NTK = 4  (16, 4)
                        registers.offset_addr.offset_addr_bf.nthw_ntk = static_cast<uint8_t>(host_parsing::IDUNthw_Ntk::IDU_16_4);
                        break;
                    default:
                        VPUX_THROW("mpe_frequent_mode {0}", static_cast<uint8_t>(mpeFrequencyMode));
                        break;
                }
            }

            // auto &wt_tensor_ref = srcInvariant.weights_data;

            // OPEN: swizzling key is not supported yet by VPUx
            registers.offset_addr.offset_addr_bf.swizzle_key = 0;
            registers.offset_addr.offset_addr_bf.wt_swizzle_key = 0;
            registers.offset_addr.offset_addr_bf.wt_swizzle_sel = 1;

            // OPEN: do not see variant.invarint field
            // variant.invariant_addr = &invariant.invariant;
            // OPEN: do not see variant.task_id

            // OPEN: skip convert relative address
            // OPEN: do not see invariant extension
            // OPEN: should I skip invariant_addr
            // variantWrapper.variant.invariant_addr = checked_cast<uint32_t>(reinterpret_cast<uint64_t>(&invariant.invariant));

            vpux::VPUIP::DPUVariantTask variantTask;
            variantTask.dpuVariantWrapper = std::move(variantWrapper);
            variantWrappers.push_back(std::move(variantTask));
        }

        // OPEN: do not see invariant task_id
        invariant.variant_count = variantWrappers.size();

        // OPEN: do not see layerOpDPU field
        // TODO does this make sense?
        invariant.cluster = parseIntArrayAttr<uint32_t>(inputOp.localeIndex())[0];
        // OPEN: do not see padding usage in runtime
        invariant.padding = 0;

        for (const auto& barrierValue : nceTask.waitBarriers()) {
            auto barrierOp = barrierValue.getDefiningOp<VPUIP::ConfigureBarrierOp>();
            VPUX_THROW_UNLESS(barrierOp, "Encountered unexpected barrier value {0} in waitBarriers range of {1}",
                            barrierValue, nceTask);

            const auto& barrier = barrierOp.barrier();
            VPUX_THROW_UNLESS(barriers.count(barrier),
                            "Encountered unpexpected {0}, all bariers must have been parsed already", barrier);
            auto& barrierWrapper = barriers[barrier];

            barrierWrapper->consumer_count++;
            invariant.invariant.barriers.consumer_mask |= 1 << barrierWrapper->real_id;
        }

        for (const auto& barrierValue : nceTask.updateBarriers()) {
            auto barrierOp = barrierValue.getDefiningOp<VPUIP::ConfigureBarrierOp>();
            VPUX_THROW_UNLESS(barrierOp, "Encountered unexpected barrier value {0} in updateBarriers range of {1}",
                            barrierValue, nceTask);

            const auto& barrier = barrierOp.barrier();
            VPUX_THROW_UNLESS(barriers.count(barrier),
                            "Encountered unpexpected {0}, all bariers must have been parsed already", barrier);
            auto& barrierWrapper = barriers[barrier];

            barrierWrapper->producer_count++;
            invariant.invariant.barriers.producer_mask |= 1 << barrierWrapper->real_id;
        }

        vpux::VPUIP::DPUTask dpuTask;
        dpuTask.dpuVariants = std::move(variantWrappers);
        dpuTask.dpuInvariant.opType = taskType;
        dpuTask.dpuInvariant.dpuInvariantWrapper = std::move(invariant);

        auto& inputPatchingInfo = dpuTask.dpuInvariant.input;
        if (auto inputDeclareOp = input.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            inputPatchingInfo.dataOffset = inputDeclareOp.dataIndex();
            inputPatchingInfo.location.memLocation = inputDeclareOp.locale();
            inputPatchingInfo.location.locationIndex = 0;
        } else if (auto inputConstDeclareOp = input.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(constantIndexes.count(input), "Encountered unexpected value {0}", inputConstDeclareOp);
            inputPatchingInfo.dataOffset = constantIndexes[input];
            inputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::GraphFile;
            inputPatchingInfo.location.locationIndex = 0;
        } else if (input.isa<mlir::BlockArgument>()) {
            inputPatchingInfo.dataOffset = 0; // dataOffset inside input
            inputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
            inputPatchingInfo.location.locationIndex = 0; // input's index
        } else {
            VPUX_THROW("Encountered unsupported defining op {0} for input of {1}", input, nceTask);
        }

        auto& weightsPatchingInfo = dpuTask.dpuInvariant.weights;
        if (auto weightsDeclareOp = weights.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            weightsPatchingInfo.dataOffset = weightsDeclareOp.dataIndex();
            weightsPatchingInfo.location.memLocation = weightsDeclareOp.locale();
            weightsPatchingInfo.location.locationIndex = 0;
        } else if (auto weightsConstDeclareOp = weights.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(constantIndexes.count(weights), "Encountered unexpected value {0}", weightsConstDeclareOp);
            weightsPatchingInfo.dataOffset = constantIndexes[weights];
            weightsPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::GraphFile;
            weightsPatchingInfo.location.locationIndex = 0;
        } else if (weights.isa<mlir::BlockArgument>()) {
            weightsPatchingInfo.dataOffset = 0; // dataOffset inside weights
            weightsPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
            weightsPatchingInfo.location.locationIndex = 0; // weights's index
        } else {
            VPUX_THROW("Encountered unsupported defining op {0} for weights of {1}", weights, nceTask);
        }

        auto weightsTable = nceTask.weight_table();
        auto& weightsTablePatchingInfo = dpuTask.dpuInvariant.weightsTable;
        if (auto weightsTableDeclareOp = weightsTable.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            weightsTablePatchingInfo.dataOffset = weightsTableDeclareOp.dataIndex();
            weightsTablePatchingInfo.location.memLocation = weightsTableDeclareOp.locale();
            weightsTablePatchingInfo.location.locationIndex = 0;
        } else if (auto weightsTableConstDeclareOp = weightsTable.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(constantIndexes.count(weightsTable), "Encountered unexpected value {0}", weightsTableConstDeclareOp);
            weightsTablePatchingInfo.dataOffset = constantIndexes[weightsTable];
            weightsTablePatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::GraphFile;
            weightsTablePatchingInfo.location.locationIndex = 0;
        } else if (weightsTable.isa<mlir::BlockArgument>()) {
            weightsTablePatchingInfo.dataOffset = 0; // dataOffset inside weightsTable
            weightsTablePatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
            weightsTablePatchingInfo.location.locationIndex = 0; // weightsTable's index
        } else {
            VPUX_THROW("Encountered unsupported defining op {0} for weightsTable of {1}", weightsTable, nceTask);
        }

        auto& outputPatchingInfo = dpuTask.dpuInvariant.output;
        if (auto outputDeclareOp = output.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            outputPatchingInfo.dataOffset = outputDeclareOp.dataIndex();
            outputPatchingInfo.location.memLocation = outputDeclareOp.locale();
            outputPatchingInfo.location.locationIndex = 0;
        } else if (output.isa<mlir::BlockArgument>()) {
            outputPatchingInfo.dataOffset = 0; // dataOffset inside output
            outputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableOutput;
            outputPatchingInfo.location.locationIndex = 0; // output's index
        } else {
            VPUX_THROW("Encountered unsupported defining op for output of {0}", nceTask);
        }
    });

    VPUX_THROW_UNLESS(!leadingDmaTasks.empty(), "Expected to encounter at least one leading DMA task");
    for (auto dma : vpux::irange(static_cast<size_t>(0), leadingDmaTasks.size() - 1)) {
        leadingDmaTasks[dma].linkAddress.metaDataLocation =
                vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::DDR_DMA;
        leadingDmaTasks[dma].linkAddress.dmaTaskIndex = dma + 1;
    }

    leadingDmaTasks.back().linkAddress.metaDataLocation = vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::NONE;

    if (!dmaTasks.empty()) {
        for (auto dma : vpux::irange(static_cast<size_t>(0), dmaTasks.size() - 1)) {
            dmaTasks[dma].linkAddress.metaDataLocation =
                    vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::RTM_DMA;
            dmaTasks[dma].linkAddress.dmaTaskIndex = dma + 1;
        }

        dmaTasks.back().linkAddress.metaDataLocation = vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::NONE;
    }

    llvm::SmallVector<mlir::ShapedType> inputs;
    for (auto inputInfo : netOp.getInputsInfo()) {
        const auto userType = inputInfo.userType().dyn_cast_or_null<mlir::ShapedType>();
        VPUX_THROW_UNLESS(userType, "Encountered unknown input value {0}", inputInfo);
        inputs.emplace_back(userType);
    }

    llvm::SmallVector<mlir::ShapedType> outputs;
    for (auto outputInfo : netOp.getOutputsInfo()) {
        const auto userType = outputInfo.userType().dyn_cast_or_null<mlir::ShapedType>();
        VPUX_THROW_UNLESS(userType, "Encountered unknown output value {0}", outputInfo);
        outputs.emplace_back(userType);
    }

    vpux::VPUIP::ELFBlobSerializer serializer;
    serializer.setNetworkInputs(inputs);
    serializer.setNetworkOutputs(outputs);

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    auto memAttr = VPUIP::PhysicalMemoryAttr::get(module->getContext(), VPUIP::PhysicalMemory::DDR);
    auto ddrResources = resources.getUsedMemory(memAttr);

    serializer.setDDRScratch(ddrResources ? ddrResources.size().count() : 0);
    serializer.setResourceRequirements(resourceRequirements);
    serializer.setLeadingDMACount(leadingDmaTasks.size());

    leadingDmaTasks.reserve(leadingDmaTasks.size() + dmaTasks.size());
    std::move(dmaTasks.begin(), dmaTasks.end(), std::back_inserter(leadingDmaTasks));
    serializer.setDMATasks(llvm::makeArrayRef(leadingDmaTasks));

    {
        std::vector<host_parsing::BarrierWrapper> barriers;
        barriers.reserve(barriersList.size());
        std::move(barriersList.begin(), barriersList.end(), std::back_inserter(barriers));
        serializer.setBarrierConfigs(llvm::makeArrayRef(barriers));
    }
    serializer.setDPUTasks(llvm::makeArrayRef(dpuTasks));
    serializer.setConstData(llvm::makeArrayRef(constants));

    return serializer.getBlob();
}

}  // namespace

std::vector<char> vpux::VPUIP::exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming, Logger log, const Config* config) {
    if (config == nullptr) {
        return exportToBlobGraphFile(module, rootTiming, log);
    }

    auto blobFormat = config->get<BLOB_FORMAT>();
    switch (blobFormat) {
    case InferenceEngine::VPUXConfigParams::BlobFormat::GRAPH_FILE:
        return exportToBlobGraphFile(module, rootTiming, log);
    case InferenceEngine::VPUXConfigParams::BlobFormat::ELF:
        return exportToBlobELF(module, rootTiming, log);
    default:
        VPUX_THROW("Unsupported blob format");
    }
}
