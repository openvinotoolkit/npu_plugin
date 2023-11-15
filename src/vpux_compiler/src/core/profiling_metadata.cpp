//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Visitors.h>

#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/profiling/generated/schema/profiling_generated.h"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include "vpux/utils/core/profiling.hpp"

using namespace vpux;

namespace {

using BarrierMap = DenseMap<mlir::Value, uint32_t>;
using TaskBarriers = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

template <class BarrierOp>
BarrierMap getBarriers(mlir::func::FuncOp funcOp) {
    BarrierMap barriersIds;
    for (BarrierOp barrierOp : funcOp.getOps<BarrierOp>()) {
        auto val = barrierOp.getBarrier();
        VPUX_THROW_UNLESS(barriersIds.count(val) == 0, "Value {0} was already serialized", val);
        barriersIds.insert({val, checked_cast<uint32_t>(barriersIds.size())});
    }
    return barriersIds;
}

flatbuffers::Offset<ProfilingFB::DPUTask> createDPUTaskMeta(flatbuffers::FlatBufferBuilder& builder,
                                                            VPUIP::DpuProfilingMetadataAttr metaAttr,
                                                            const std::string& name,
                                                            const std::vector<uint32_t>& waitBarriers,
                                                            const std::vector<uint32_t>& updateBarriers,
                                                            const std::vector<uint32_t>& workloadIds) {
    VPUX_THROW_WHEN(metaAttr.getNumVariants() == nullptr, "Missed numVariants information for DpuMetaSerialization");
    VPUX_THROW_WHEN(metaAttr.getClusterId() == nullptr, "Missed clusterId information for DpuMetaSerialization");

    const auto bufferId = metaAttr.getBufferId().getInt();
    const auto clusterId = metaAttr.getClusterId().getInt();
    const auto taskId = metaAttr.getTaskId().getInt();
    const auto numVariants = metaAttr.getNumVariants().getInt();
    const auto maxVariants = metaAttr.getMaxVariants().getInt();

    VPUX_THROW_WHEN(workloadIds.size() != 0 && workloadIds.size() != static_cast<size_t>(numVariants),
                    "Expected {0} workloads, but got {1}", numVariants, workloadIds.size());

    const auto nameOffset = builder.CreateString(name);
    const auto waitBarriersOffset = builder.CreateVector(waitBarriers);
    const auto updateBarriersOffset = builder.CreateVector(updateBarriers);
    const auto workloadIdsOffset = builder.CreateVector(workloadIds);

    return ProfilingFB::CreateDPUTask(builder, nameOffset, bufferId, clusterId, taskId, numVariants, maxVariants,
                                      waitBarriersOffset, updateBarriersOffset, workloadIdsOffset);
}

size_t getSectionTotalSize(IE::DataInfoOp profilingOutputInfo) {
    const auto profilingType = profilingOutputInfo.userType().cast<vpux::NDTypeInterface>();
    const auto shape = profilingType.getShape();

    VPUX_THROW_WHEN(shape.size() != 1, "Invalid profiling output shape '{0}'. Must be 1D tensor");
    VPUX_THROW_UNLESS(profilingType.getElementType().isInteger(CHAR_BIT * sizeof(uint32_t)),
                      "Profiling tensor type must be ui32, but got '{0}'", profilingType.getElementType());

    const uint32_t totalSize = shape[DimsOrder::C.dimAt(0)] * sizeof(uint32_t);
    return totalSize;
}

TaskBarriers getOpBarriersImpl(const BarrierMap& virtBarriers, const mlir::ValueRange waitBarriers,
                               const mlir::ValueRange updateBarriers) {
    const auto extractBarriersIDs = [&virtBarriers](const mlir::ValueRange barriers) -> std::vector<uint32_t> {
        std::vector<uint32_t> ids;
        ids.reserve(barriers.size());
        for (const auto& bar : barriers) {
            const auto it = virtBarriers.find(bar);
            VPUX_THROW_UNLESS(it != virtBarriers.end(), "Value {0} wasn't serialized yet", bar);
            ids.push_back(it->second);
        }
        return ids;
    };

    std::vector<uint32_t> waitIds = extractBarriersIDs(waitBarriers);
    std::vector<uint32_t> updateIds = extractBarriersIDs(updateBarriers);

    return std::make_pair(waitIds, updateIds);
}

struct MiDialectProvider {
    template <class TargetOp>
    static auto extractOp(mlir::func::FuncOp funcOp) {
        return funcOp.getOps<TargetOp>();
    }

    template <class TargetOp>
    static auto extractSwOp(mlir::func::FuncOp funcOp) {
        return extractOp<TargetOp>(funcOp);
    }

    template <class OpType>
    static TaskBarriers getOpBarriers(const BarrierMap& virtBarriers, OpType op) {
        return getOpBarriersImpl(virtBarriers, op.getWaitBarriers(), op.getUpdateBarriers());
    }

    static std::vector<uint32_t> getWorkloadIds(VPUMI37XX::DPUInvariantOp dpuInvariant) {
        std::vector<uint32_t> workloadIds;
        for (const auto& user : dpuInvariant->getUsers()) {
            if (auto dpuVariant = mlir::dyn_cast_or_null<VPUMI37XX::DPUVariantOp>(user)) {
                if (dpuVariant.getWorkloadId().has_value()) {
                    workloadIds.push_back(dpuVariant.getWorkloadId().value());
                }
            }
        }
        std::reverse(workloadIds.begin(), workloadIds.end());
        return workloadIds;
    }

    template <class... Args>
    static unsigned short getDmaHwpId(Args...) {
        return 0;
    }
};

struct RtDialectProvider {
    template <class TargetOp>
    static auto extractOp(mlir::func::FuncOp funcOp) {
        SmallVector<TargetOp> ops;
        funcOp->walk([&](VPURT::TaskOp taskOp) {
            if (auto innerOp = mlir::dyn_cast<TargetOp>(taskOp.getInnerTaskOp())) {
                ops.push_back(innerOp);
            }
        });
        return ops;
    }

    static TaskBarriers getOpBarriers(const BarrierMap& virtBarriers, mlir::Operation* op) {
        auto parentOp = op->getParentOfType<VPURT::TaskOp>();
        VPUX_THROW_WHEN(op == nullptr, "Parent must be VPURT::TaskOp");
        return getOpBarriersImpl(virtBarriers, parentOp.getWaitBarriers(), parentOp.getUpdateBarriers());
    }

    template <class TargetOp>
    static auto extractSwOp(mlir::func::FuncOp funcOp) {
        return extractOp<TargetOp>(funcOp);
    }

    static std::vector<uint32_t> getWorkloadIds(VPUIP::NCEClusterTaskOp dpuOp) {
        std::vector<uint32_t> workloadIds;
        for (VPUIP::DPUTaskOp variant : dpuOp.variants().getOps<VPUIP::DPUTaskOp>()) {
            if (variant.workload_id().has_value()) {
                workloadIds.push_back(variant.workload_id().value());
            }
        }
        return workloadIds;
    }

    static unsigned short getDmaHwpId(VPUIP::DMATypeOpInterface dmaOp) {
        if (auto hwpIdAttr = dmaOp.getDmaHwpIdAttr()) {
            return static_cast<unsigned short>(hwpIdAttr.getSInt());
        }
        return 0;
    }
};

struct RtDialectProvider30XX : public RtDialectProvider {
    template <class... Args>
    static auto extractSwOp(mlir::func::FuncOp funcOp) {
        SmallVector<mlir::Operation*> ops;
        funcOp->walk([&](VPURT::TaskOp taskOp) {
            if (taskOp.getExecutorKind() == vpux::VPU::ExecutorKind::SHAVE_UPA) {
                ops.push_back(taskOp.getInnerTaskOp());
            }
        });
        return ops;
    }
};

struct DummyUpaOp {
    static StringRef getOperationName() {
        return "";
    }
};

template <typename TaskType>
using FbVector = flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TaskType>>>;

std::string cleanSwTaskType(std::string origType) {
    const std::vector<std::pair<std::string, std::string>> REPLACE_PAIRS = {{"VPUIP.", ""}, {"UPA", ""}};
    return std::accumulate(REPLACE_PAIRS.cbegin(), REPLACE_PAIRS.cend(), std::move(origType),
                           [](std::string a, const auto& replacement) {
                               const auto pos = a.find(replacement.first);
                               if (pos == std::string::npos) {
                                   return a;
                               }
                               return a.replace(pos, replacement.first.size(), replacement.second);
                           });
}

template <class DialectProvider, class DmaType, class Iterable>
FbVector<ProfilingFB::DMATask> getDmaTasksOffset(flatbuffers::FlatBufferBuilder& builder, const Iterable& dmaTasks,
                                                 const BarrierMap& barriers) {
    std::vector<flatbuffers::Offset<ProfilingFB::DMATask>> dmaOffsets;
    for (const auto& dmaTask : dmaTasks) {
        const auto name = stringifyLocation(dmaTask->getLoc());
        const unsigned short hwpId = DialectProvider::getDmaHwpId(dmaTask);

        const mlir::Type inputType = const_cast<DmaType&>(dmaTask).getInput().getType();
        const auto ndType = inputType.cast<vpux::NDTypeInterface>();
        const auto sourceLocale = VPU::stringifyMemoryKind(ndType.getMemoryKind()).str();

        const auto opBarriers = DialectProvider::getOpBarriers(barriers, dmaTask);
        const auto nameOffset = builder.CreateString(name);
        const auto sourceLocaleOffset = builder.CreateString(sourceLocale);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);
        const auto taskOffset = ProfilingFB::CreateDMATask(builder, nameOffset, sourceLocaleOffset, waitBarriersOffset,
                                                           updateBarriersOffset, hwpId);
        dmaOffsets.push_back(taskOffset);
    }
    return builder.CreateVector(dmaOffsets);
}

template <class DialectProvider, class DpuType, class Iterable>
FbVector<ProfilingFB::DPUTask> getDpuTasksOffset(flatbuffers::FlatBufferBuilder& builder, const Iterable& dpuTasks,
                                                 const BarrierMap& barriers) {
    std::vector<flatbuffers::Offset<ProfilingFB::DPUTask>> dpuOffsets;
    for (const auto& dpuInvariant : dpuTasks) {
        auto name = stringifyLocation(dpuInvariant->getLoc());

        const auto opBarriers = DialectProvider::getOpBarriers(barriers, dpuInvariant);
        std::vector<uint32_t> workloadIds = DialectProvider::getWorkloadIds(dpuInvariant);

        // TableGen generate interface methods without const specifier, so can't be called from const DpuType&.
        // In the same moment, coverity force to use const auto&
        auto profMeta = const_cast<DpuType&>(dpuInvariant).getProfilingMetadata();
        VPUX_THROW_UNLESS(profMeta.has_value(), "Empty profiling metadata at '{0}'", dpuInvariant);

        const auto taskOffset =
                createDPUTaskMeta(builder, profMeta.value(), name, opBarriers.first, opBarriers.second, workloadIds);
        dpuOffsets.push_back(taskOffset);
    }
    return builder.CreateVector(dpuOffsets);
}

template <class DialectProvider, class SwType, class Iterable>
FbVector<ProfilingFB::SWTask> getSwTasksOffset(flatbuffers::FlatBufferBuilder& builder, const Iterable& swTasks,
                                               const BarrierMap& barriers) {
    std::vector<flatbuffers::Offset<ProfilingFB::SWTask>> swTaskOffsets;
    for (const auto& swTask : swTasks) {
        auto name = stringifyLocation(swTask->getLoc());
        std::string swTaskType;
        // ActShave store kernel as attribute, so for all task same operation used. In case of UPA for each operation
        // new op added to dialect
        const auto taskType = swTask->getName().getStringRef().str();
        if (SwType::getOperationName().str() != taskType) {
            swTaskType = cleanSwTaskType(taskType);
        }

        const auto opBarriers = DialectProvider::getOpBarriers(barriers, swTask);

        const auto nameOffset = builder.CreateString(name);
        const auto typeOffset = builder.CreateString(swTaskType);
        const auto waitBarriersOffset = builder.CreateVector(opBarriers.first);
        const auto updateBarriersOffset = builder.CreateVector(opBarriers.second);
        const auto taskOffset =
                ProfilingFB::CreateSWTask(builder, nameOffset, waitBarriersOffset, updateBarriersOffset, typeOffset);

        swTaskOffsets.push_back(taskOffset);
    }
    return builder.CreateVector(swTaskOffsets);
}

flatbuffers::Offset<ProfilingFB::ProfilingBuffer> createProfilingBufferOffset(flatbuffers::FlatBufferBuilder& builder,
                                                                              IE::CNNNetworkOp netOp,
                                                                              mlir::func::FuncOp) {
    auto profilingOutputsInfo = netOp.getProfilingOutputsInfo();
    VPUX_THROW_WHEN(profilingOutputsInfo.size() != 1, "Unexpected number of profiling outputs (expected 1, got {0})",
                    profilingOutputsInfo.size());

    IE::DataInfoOp profilingOutputInfo = *profilingOutputsInfo.front().getOps<IE::DataInfoOp>().begin();
    auto& sections = profilingOutputInfo.sections().front().front();

    std::vector<flatbuffers::Offset<ProfilingFB::ProfilingSection>> profilingSectionsOffsets;
    for (auto section : sections.getOps<VPUIP::ProfilingSectionOp>()) {
        const auto secType = section.sectionType();
        const auto offset = section.offset();
        const auto size = section.size();

        const auto sectionOffset = ProfilingFB::CreateProfilingSection(builder, secType, offset, size);
        profilingSectionsOffsets.push_back(sectionOffset);
    }

    const auto sectionsOffset = builder.CreateVector(profilingSectionsOffsets);
    const auto sectionTotalSizeBytes = getSectionTotalSize(profilingOutputInfo);
    return ProfilingFB::CreateProfilingBuffer(builder, sectionsOffset, sectionTotalSizeBytes);
}

template <typename DialectProvider, class BarrierOp, class DmaType, class DpuType, class SwType>
flatbuffers::DetachedBuffer buildProfilingMetaGeneric(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger log) {
    log.trace("building Profiling Metadata");

    flatbuffers::FlatBufferBuilder builder;

    const auto barriers = getBarriers<BarrierOp>(funcOp);

    auto dmaOffset = getDmaTasksOffset<DialectProvider, DmaType>(
            builder, DialectProvider::template extractOp<DmaType>(funcOp), barriers);
    auto dpuOffset = getDpuTasksOffset<DialectProvider, DpuType>(
            builder, DialectProvider::template extractOp<DpuType>(funcOp), barriers);
    auto swTaskOffset = getSwTasksOffset<DialectProvider, SwType>(
            builder, DialectProvider::template extractSwOp<SwType>(funcOp), barriers);
    auto profilingBufferOffset = createProfilingBufferOffset(builder, netOp, funcOp);

    auto metadataOffset =
            ProfilingFB::CreateProfilingMeta(builder, profilingBufferOffset, dmaOffset, dpuOffset, swTaskOffset);
    builder.Finish(metadataOffset);

    return builder.Release();
}

flatbuffers::DetachedBuffer buildProfilingMetaVPURT30XX(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger log) {
    return buildProfilingMetaGeneric<RtDialectProvider30XX, VPURT::ConfigureBarrierOp, VPUIP::DMATypeOpInterface,
                                     VPUIP::NCEClusterTaskOp, DummyUpaOp>(netOp, funcOp, log);
}

flatbuffers::DetachedBuffer buildProfilingMetaVPURTGeneral(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp,
                                                           Logger log) {
    return buildProfilingMetaGeneric<RtDialectProvider, VPURT::ConfigureBarrierOp, VPUIP::DMATypeOpInterface,
                                     VPUIP::NCEClusterTaskOp, VPUIP::SwKernelOp>(netOp, funcOp, log);
}
};  // namespace

flatbuffers::DetachedBuffer vpux::buildProfilingMetaMI37XX(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp,
                                                           Logger log) {
    return buildProfilingMetaGeneric<MiDialectProvider, VPUMI37XX::ConfigureBarrierOp, VPUMI37XX::NNDMAOp,
                                     VPUMI37XX::DPUInvariantOp, VPUMI37XX::ActKernelInvocationOp>(netOp, funcOp, log);
}

flatbuffers::DetachedBuffer vpux::buildProfilingMetaVPURT(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp,
                                                          Logger log) {
    const auto arch = VPU::getArch(funcOp);
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return ::buildProfilingMetaVPURT30XX(netOp, funcOp, log);
    case VPU::ArchKind::VPUX37XX:
        return ::buildProfilingMetaVPURTGeneral(netOp, funcOp, log);
    default:
        VPUX_THROW("Unknown architecture: {0}", arch);
    }
}
