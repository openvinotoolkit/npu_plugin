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

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/backend/VPUIP.hpp"

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

#include <unordered_map>
#include <deque>

#include "host_parsed_inference.h"
#include "nce_2p7_hw.h"

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

std::vector<char> exportToBlobELF(mlir::ModuleOp module, mlir::TimingScope& rootTiming, Logger log) {
    std::cerr << "ELF" << '\n';
    log.setName("VPUIP::BackEnd (ELF)");

    log.trace("Extract 'IE.{0}' from Module (ELF)", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Extract 'VPUIP.{0}' from Module (ELF)", VPUIP::GraphOp::getOperationName());
    auto graphOp = VPUIP::GraphOp::getFromModule(module);

    std::vector<BarrierWrapper> barriersList;
    mlir::DenseMap<mlir::Value, std::vector<BarrierWrapper>::iterator> barriers;
    netFunc.walk([&barriersList, &barriers](VPUIP::ConfigureBarrierOp barrierOp) {
        BarrierWrapper wrapper{};
        wrapper.real_id = checked_cast<uint8_t>(barrierOp.id());
        VPUX_THROW_UNLESS(barriers.try_emplace(barrierOp.barrier(),
                                               barriersList.insert(barriersList.end(), std::move(wrapper))).second,
                          "Encountered the same barrierOp {0} in function twice", barrierOp);
    });

    VPUX_THROW_UNLESS(barriersList.size() <= 32, "Networks with barrriers count more than 32 are not supported");

    for (auto barriersPosition = barriersList.begin(); barriersPosition != barriersList.end(); ++barriersPosition) {
        const auto nextSameId = std::find_if(std::next(barriersPosition), barriersList.end(),
                                             [&barriersPosition](const auto& barrier) {
                                                 return barriersPosition->real_id == barrier.real_id;
                                             });
        barriersPosition->next_same_id = checked_cast<int32_t>(std::distance(nextSameId, barriersList.end()));
    }

    HostParsedInference hostParsedInference{};
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

            VPUX_THROW_UNLESS(resourceRequirements.slice_count == 0, "Encountered more than one resource of kind {0}", kind);
            resourceRequirements.slice_count = checked_cast<uint8_t>(res.count());
        });
    }

    mlir::DenseMap<mlir::Value, std::size_t> constantIndexes;
    std::size_t currentIndex = 0;
    netFunc.walk([&constantIndexes, &currentIndex](vpux::Const::DeclareOp constant) {
        auto const insertion = constantIndexes.insert({constant.output(), currentIndex});
        if (insertion.second) {
            currentIndex++;
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

        auto const inputShapeLogical = vpux::getShape(inputType);
        auto const inputStridesLogical = vpux::getStrides(inputType);
        auto const inputOrder = vpux::DimsOrder::fromType(inputType);
        auto const inputShapeMemory = inputOrder.toMemoryOrder(inputShapeLogical);
        auto const inputStridesMemory = inputOrder.toMemoryOrder(inputStridesLogical);

        auto const outputShapeLogical = vpux::getShape(outputType);
        auto const outputStridesLogical = vpux::getStrides(outputType);
        auto const outputOrder = vpux::DimsOrder::fromType(outputType);
        auto const outputShapeMemory = outputOrder.toMemoryOrder(outputShapeLogical);
        auto const outputStridesMemory = outputOrder.toMemoryOrder(outputStridesLogical);

        vpux::VPUIP::DmaTask task{};
        auto& dmaDescriptor = task.dmaDescriptor.transaction;
        dmaDescriptor.cfg_link.cfg_bits.type         = 1;
        dmaDescriptor.cfg_link.cfg_bits.burst_length = 16;
        dmaDescriptor.cfg_link.cfg_bits.critical     = 1;
        dmaDescriptor.cfg_link.cfg_bits.barrier_en   = 1;
        dmaDescriptor.length                         = inputType.getNumElements() * vpux::Byte(vpux::getElemTypeSize(inputType)).count();
        dmaDescriptor.num_planes                     = inputShapeLogical[vpux::Dims4D::Act::C];
        dmaDescriptor.src_plane_stride               = vpux::Byte(inputStridesMemory[vpux::MemDim(vpux::Dims4D::Act::C.ind())]).count();
        dmaDescriptor.dst_plane_stride               = vpux::Byte(outputStridesMemory[vpux::MemDim(vpux::Dims4D::Act::C.ind())]).count();

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
            auto barrierWrapper = barriers[barrier];

            barrierWrapper->consumer_count++;
            dmaDescriptor.barriers.prod_mask |= 1 << barrierWrapper->real_id;
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
            dmaDescriptor.barriers.cons_mask |= 1 << barrierWrapper->real_id;
        }

        auto& inputPatchingInfo = task.input;
        if (auto inputDeclareOp = input.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            inputPatchingInfo.dataOffset = inputDeclareOp.dataIndex();
            inputPatchingInfo.location.memLocation = inputDeclareOp.locale();
            inputPatchingInfo.location.locationIndex = 0;
        } else if (auto inputConstDeclareOp = input.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(constantIndexes.count(input), "Encountered unexpected value {0}", inputConstDeclareOp);
            inputPatchingInfo.dataOffset = 0;
            inputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::GraphFile;
            inputPatchingInfo.location.locationIndex = constantIndexes[input];
        } else if (input.isa<mlir::BlockArgument>()) {
            inputPatchingInfo.dataOffset = 0; // dataOffset inside input
            inputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableInput;
            inputPatchingInfo.location.locationIndex = 0; // input's index
        } else {
            VPUX_THROW("Encountered unsupported defining op {0} for input of {1}", input, dmaTask);
        }

        auto& outputPatchingInfo = task.output;
        if (auto outputDeclareOp = output.getDefiningOp<VPUIP::DeclareTensorOp>()) {
            outputPatchingInfo.dataOffset = outputDeclareOp.dataIndex();
            outputPatchingInfo.location.memLocation = outputDeclareOp.locale();
            outputPatchingInfo.location.locationIndex = 0;
        } else if (output.isa<mlir::BlockArgument>()) {
            outputPatchingInfo.dataOffset = 0; // dataOffset inside output
            outputPatchingInfo.location.memLocation = vpux::VPUIP::MemoryLocation::ProgrammableOutput;
            outputPatchingInfo.location.locationIndex = 0; // output's index
        } else {
            VPUX_THROW("Encountered unsupported defining op for output of {0}", dmaTask);
        }

        insertPosition->push_back(std::move(task));
    });

    VPUX_THROW_UNLESS(!leadingDmaTasks.empty(), "Expected to encounter at least one leading DMA task");
    for (auto dma : vpux::irange(static_cast<size_t>(0), leadingDmaTasks.size() - 1)) {
        leadingDmaTasks[dma].linkAddress.metaDataLocation = vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::DDR_DMA;
        leadingDmaTasks[dma].linkAddress.dmaTaskIndex = dma + 1;
    }

    leadingDmaTasks.back().linkAddress.metaDataLocation = vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::NONE;

    if (!dmaTasks.empty()) {
        for (auto dma : vpux::irange(static_cast<size_t>(0), dmaTasks.size() - 1)) {
            dmaTasks[dma].linkAddress.metaDataLocation = vpux::VPUIP::LinkAddressPatchingInfo::MetaDataLocation::RTM_DMA;
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
    for (auto outputInfo : netOp.getInputsInfo()) {
        const auto userType = outputInfo.userType().dyn_cast_or_null<mlir::ShapedType>();
        VPUX_THROW_UNLESS(userType, "Encountered unknown output value {0}", outputInfo);
        outputs.emplace_back(userType);
    }

    vpux::VPUIP::ELFBlobSerializer serializer;
    serializer.setNetworkInputs(inputs);
    serializer.setNetworkOutputs(outputs);

    serializer.setDDRScratch(0);
    serializer.setResourceRequirements(resourceRequirements);
    serializer.setLeadingDMACount(leadingDmaTasks.size());

    leadingDmaTasks.resize(leadingDmaTasks.size() + dmaTasks.size());
    std::move(dmaTasks.begin(), dmaTasks.end(), std::back_inserter(leadingDmaTasks));
    serializer.setDMATasks(llvm::makeArrayRef(leadingDmaTasks));
    serializer.setBarrierConfigs(llvm::makeArrayRef(barriersList));

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
