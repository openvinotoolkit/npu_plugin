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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

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

void setActivityFactor(VPUIP::PhysicalProcessor processor, MVCNN::ProcessorMappingBuilder& builder,
                       mlir::ModuleOp module) {
    // TODO: calc this value during compilation
    static const float activityFactor = 90.0;
    const auto arch = VPUIP::getArch(module);
    if (arch == VPUIP::ArchKind::KMB || arch == VPUIP::ArchKind::TBH) {
        if (processor == VPUIP::PhysicalProcessor::NCE_Cluster || processor == VPUIP::PhysicalProcessor::SHAVE_UPA) {
            builder.add_activity_factor(activityFactor);
        }
    } else if (arch == VPUIP::ArchKind::MTL) {
        if (processor == VPUIP::PhysicalProcessor::NCE_Cluster || processor == VPUIP::PhysicalProcessor::SHAVE_NN) {
            builder.add_activity_factor(activityFactor);
        }
    }
}

flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorMapping(VPUIP::BlobWriter& writer,
                                                                    IERT::ExecutorResourceOp res,
                                                                    mlir::ModuleOp module) {
    const auto kind = res.kind().dyn_cast_or_null<VPUIP::PhysicalProcessorAttr>();
    VPUX_THROW_UNLESS(kind != nullptr, "Got unknown executor kind '{0}'", res.kind());

    const auto processor = kind.getValue();
    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(processor));
    builder.add_number(checked_cast<double>(res.count()));
    builder.add_is_bitmask(false);
    setActivityFactor(processor, builder, module);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::ProcessorMapping> createProcessorFreqMapping(VPUIP::BlobWriter& writer,
                                                                        IERT::ExecutorResourceOp res) {
    const auto kind = res.kind().dyn_cast_or_null<VPUIP::PhysicalProcessorAttr>();
    VPUX_THROW_UNLESS(kind != nullptr, "Got unknown executor kind '{0}'", res.kind());

    MVCNN::ProcessorMappingBuilder builder(writer);
    builder.add_item(createPhysicalProcessor(kind.getValue()));
    builder.add_number(VPUIP::getProcessorFrequency(res));
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

flatbuffers::Offset<MVCNN::MemoryRelationshipMapping> createBandwidthMapping(VPUIP::BlobWriter& writer,
                                                                             IERT::MemoryResourceOp src,
                                                                             IERT::MemoryResourceOp dst,
                                                                             double bandwidth) {
    MVCNN::MemoryRelationshipMappingBuilder builder(writer);
    const auto srcKind = src.kindAttr().dyn_cast_or_null<VPUIP::PhysicalMemoryAttr>();
    VPUX_THROW_UNLESS(srcKind != nullptr, "Got unknown memory space kind '{0}'", src.kindAttr());
    const auto dstKind = dst.kindAttr().dyn_cast_or_null<VPUIP::PhysicalMemoryAttr>();
    VPUX_THROW_UNLESS(dstKind != nullptr, "Got unknown memory space kind '{0}'", dst.kindAttr());
    builder.add_from_item(createPhysicalMem(srcKind.getValue()));
    builder.add_to_item(createPhysicalMem(dstKind.getValue()));
    builder.add_number(bandwidth);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Resources> createResources(VPUIP::BlobWriter& writer, mlir::ModuleOp module) {
    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    VPUX_THROW_UNLESS(resources != nullptr, "Missing IERT run-time resources information");

    const auto usedMemory =
            writer.createVector(resources.getUsedMemory() | transformed([&](IERT::MemoryResourceOp res) {
                                    return createMemoryMapping(writer, res);
                                }));

    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> executorsOffsets;
    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> processorVec;
    resources.walk([&](IERT::ExecutorResourceOp res) {
        if (res.kind().isa<VPUIP::PhysicalProcessorAttr>()) {
            executorsOffsets.push_back(createProcessorMapping(writer, res, module));
            if (res->hasAttr(VPUIP::getProcessorFrequencyAttrName())) {
                processorVec.push_back(createProcessorFreqMapping(writer, res));
            }
        }
    });
    const auto executors = writer.createVector(executorsOffsets);
    const auto processorFrequency = writer.createVector(processorVec);

    SmallVector<flatbuffers::Offset<MVCNN::MemoryRelationshipMapping>> memoryVec;
    SmallVector<IERT::MemoryResourceOp> memoryTypes;
    resources.walk([&](IERT::MemoryResourceOp src) {
        if (src->hasAttr(VPUIP::getBandwidthAttrName())) {
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
    // only SW_kernel operations can generate kernelData, from either built-in functions or from custom
    auto kernelGenOps = to_small_vector(netFunc.getOps<VPUIP::SW_Kernel>());
    if (kernelGenOps.empty()) {
        return {};
    }

    // TODO: extract num shaves info from IERT::RuntimeResourcesOp, which can be extracted from module
    const long int maxShaves = 4;

    const auto stack_size{1U << 12};  // 4KB stack

    llvm::SmallVector<uint8_t, stack_size> shave_stack_data(stack_size);
    std::vector<flatbuffers::Offset<MVCNN::KernelDataReference>> stacks(maxShaves);  // 4 Activation SHAVEs for MTL

    for (uint32_t shv{}; shv < maxShaves; ++shv) {
        log.trace("act-shave {0}_stack size is {1}", shv, stack_size);

        stacks[shv] = writer.createKernelDataRef("actSHAVE" + std::to_string(shv) + "_stack",
                                                 vpux::VPUIP::MemoryLocation::GFEmbeddedKernel, 0, stack_size,
                                                 shave_stack_data);
    }

    const auto stackBuffers = writer.createVector(stacks);

    llvm::SmallVector<uint8_t, 1024 + (1U << 16)> scratch_buffer(1024 +
                                                                 (1U << 16));  // 64KB scratch buffer + 1024 to align

    const uint64_t non_empty_offset = 1;
    const auto scratchBuffer =
            writer.createKernelDataRef("scratch_buffer", vpux::VPUIP::MemoryLocation::GFEmbeddedKernel,
                                       non_empty_offset, scratch_buffer.size() - 1024, scratch_buffer);

    // TODO : check that arguments in given function
    auto actKernelDesc = writer.compileManagementKernelData();

    // this is the only supported storage so far
    const auto kernelStorageLocale = vpux::VPUIP::MemoryLocation::GFEmbeddedKernel;

    auto kernelText = writer.createKernelDataRef(actKernelDesc.text, kernelStorageLocale);
    auto kernelData = writer.createKernelDataRef(actKernelDesc.data, kernelStorageLocale);

    MVCNN::ActKernelBuilder kernelbuilder(writer);
    kernelbuilder.add_kernelText(kernelText);
    kernelbuilder.add_type(MVCNN::ActKernelType_KERNEL);
    kernelbuilder.add_kernelEntry(0);
    kernelbuilder.add_globalArgs(kernelData);

    auto kernel = kernelbuilder.Finish();

    MVCNN::ActKernelRuntimeBuilder builder(writer);
    builder.add_shaveStacks(stackBuffers);
    builder.add_codeScratchBuffer(scratchBuffer);
    builder.add_kernel(kernel);

    return builder.Finish();
}

MVCNN::TargetDevice mapTargetDevice(const VPUIP::ArchKind kind) {
    switch (kind) {
    case VPUIP::ArchKind::KMB:
        return MVCNN::TargetDevice::TargetDevice_KMB;
    case VPUIP::ArchKind::TBH:
        return MVCNN::TargetDevice::TargetDevice_TBH;
    case VPUIP::ArchKind::MTL:
        return MVCNN::TargetDevice::TargetDevice_MTL;
    case VPUIP::ArchKind::LNL:
        return MVCNN::TargetDevice::TargetDevice_LNL;
    default:
        VPUX_THROW("Unsupported architecture '{0}'", kind);
    }
}

MVCNN::TargetDeviceRevision mapTargetDeviceRevision(const VPUIP::ArchKind kind) {
    switch (kind) {
    case VPUIP::ArchKind::KMB:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0;
    default:
        return MVCNN::TargetDeviceRevision::TargetDeviceRevision_NONE;
    }
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(VPUIP::BlobWriter& writer, mlir::ModuleOp module,
                                                              VPUIP::GraphOp graphOp, IE::CNNNetworkOp netOp,
                                                              mlir::FuncOp netFunc, mlir::TimingScope& rootTiming,
                                                              Logger log) {
    auto scopeTiming = rootTiming.nest("Create summary header");

    const auto allTasks = netFunc.getOps<VPUIP::TaskOpInterface>();
    const auto taskCount = std::distance(allTasks.begin(), allTasks.end());

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
    const auto serializedGraphProfilingOutputs = writer.createVector(graphProfilingOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources = createResources(writer, module);
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
    builder.add_device(mapTargetDevice(VPUIP::getArch(module)));
    builder.add_device_revision(mapTargetDeviceRevision(VPUIP::getArch(module)));
    builder.add_act_kernel_runtime(serializedActKernelsRuntime);
    return builder.Finish();
}

void serializeTensorDecls(VPUIP::BlobWriter& writer, mlir::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Serialize tensor declarations");

    size_t tempTensorInd = 0;
    netFunc.walk([&](VPUIP::DeclareTensorOp tensorOp) {
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
                                                              VPUIP::GraphOp graphOp, mlir::TimingScope& rootTiming,
                                                              Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize virtual barriers");

    SmallVector<VPUIP::BlobWriter::Barrier> virtBarriers;

    netFunc.walk([&](VPUIP::DeclareVirtualBarrierOp barrierOp) {
        log.trace("Got virtual varrier at '{0}'", barrierOp->getLoc());

        VPUX_THROW_UNLESS(VPUIP::bitEnumContains(graphOp.options(), VPUIP::ExecutionFlag::DynamicBarriers),
                          "Graph was not configured for virtual barriers usage");

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

    netFunc.walk([&](VPUIP::TaskOpInterface taskOp) {
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
                                                      Logger log) {
    log.setName("VPUIP::BackEnd");

    log.trace("Extract 'IE.{0}' from Module", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Extract 'VPUIP.{0}' from Module", VPUIP::GraphOp::getOperationName());
    auto graphOp = VPUIP::GraphOp::getFromModule(module);

    VPUIP::BlobWriter writer(log.nest());

    const auto header = createSummaryHeader(writer, module, graphOp, netOp, netFunc, rootTiming, log);

    serializeTensorDecls(writer, netFunc, rootTiming);
    const auto binaryData = serializeBinaryData(writer, netFunc, rootTiming, log);
    const auto virtBarriers = serializeVirtBarriers(writer, netFunc, graphOp, rootTiming, log);
    const auto taskLists = serializeTaskLists(writer, netFunc, rootTiming, log);
    const auto kernelData = serializeKernelData(writer, netFunc, rootTiming, log);
    const auto graphFile = createGraphFile(writer, header, taskLists, binaryData, kernelData, virtBarriers, rootTiming);

    auto finalTiming = rootTiming.nest("Finalize serialized graph");
    writer.impl().Finish(graphFile, "BLOB");
    auto detached = writer.impl().Release();

    auto serializedGraphFile = MVCNN::GetGraphFile(detached.data());

    // align KernelData section referenced by given KernelDataReference
    // returns moved offset
    auto alignKernelDataSection = [&](const MVCNN::KernelDataReference* section, auto sectionLogical) {
        auto section_data = serializedGraphFile->kernel_data()->Get(section->locale_offset())->data();

        auto offset = section_data->Data() - detached.data();
        log.trace("offset to kernel {0} {1} in Finished FBB is {2}", section->name()->c_str(), sectionLogical, offset);

        //  align calculations
        const uint32_t kilobyteAlignment = 1024;

        auto aligned_offset = llvm::alignTo(offset, kilobyteAlignment);
        offset = aligned_offset - offset;
        log.trace("move kernel {0} {1} by {2} bytes to be {3}", section->name()->c_str(), sectionLogical, offset,
                  aligned_offset);

        memmove(const_cast<uint8_t*>(section_data->Data() + offset), section_data->Data(),
                section_data->Length() - kilobyteAlignment);

        // clear beginning
        memset(const_cast<uint8_t*>(section_data->Data()), 0, offset);

        return offset;
    };

    auto alignReferenceSection = [&](const MVCNN::KernelDataReference* section, uint64_t offset) {
        // correcting data offset for section in schema
        auto table = reinterpret_cast<flatbuffers::Table*>(const_cast<MVCNN::KernelDataReference*>(section));

        const uint64_t non_empty_offset = 1;
        // updating offset pointer
        table->SetField(MVCNN::KernelDataReference::VT_DATA_OFFSET,
                        checked_cast<uint32_t>(section->data_offset() + offset - non_empty_offset), 0u);
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

                // scratchBuffer aligning
                auto scratchBuffer = serializedGraphFile->header()->act_kernel_runtime()->codeScratchBuffer();
                alignSection(scratchBuffer, ".scratchBuffer");
            }
        }
    }

    return detached;
}
