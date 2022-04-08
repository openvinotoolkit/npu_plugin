//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/core/overlap_DPU_and_DMA.hpp"
#include "vpux/compiler/core/schedule_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <file_utils.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include "vpux/compiler/core/developer_build_utils.hpp"

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

using namespace vpux;

namespace {

//
// MemRefAllocRewrite
//

class MemRefAllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    MemRefAllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult MemRefAllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.memref();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto valType = val.getType().cast<vpux::NDTypeInterface>();
    auto section = VPURT::getBufferSection(valType.getMemoryKind());
    auto sectionIndex = valType.getMemSpace().getIndex();

    if (sectionIndex.hasValue()) {
        auto sectionIndexArrayAttr = getIntArrayAttr(getContext(), makeArrayRef(sectionIndex.getValue()));
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, sectionIndexArrayAttr,
                                                            offset, nullptr);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, offset);
    }

    return mlir::success();
}

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<VPURT::Alloc> {
public:
    AllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::Alloc>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::Alloc origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(VPURT::Alloc origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.buffer();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto valType = val.getType().cast<vpux::NDTypeInterface>();
    auto section = VPURT::getBufferSection(valType.getMemoryKind());
    auto sectionIndex = valType.getMemSpace().getIndex();

    auto swizzlingKey = origOp.swizzlingKeyAttr();

    if (sectionIndex.hasValue()) {
        auto sectionIndexArrayAttr = getIntArrayAttr(getContext(), makeArrayRef(sectionIndex.getValue()));
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, sectionIndexArrayAttr,
                                                            offset, swizzlingKey);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, nullptr, offset,
                                                            swizzlingKey);
    }

    return mlir::success();
}

//
// AllocDistributedRewrite
//

class AllocDistributedRewrite final : public mlir::OpRewritePattern<VPURT::AllocDistributed> {
public:
    AllocDistributedRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURT::AllocDistributed>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::AllocDistributed origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocDistributedRewrite::matchAndRewrite(VPURT::AllocDistributed origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp->getLoc());

    const auto val = origOp.buffer();

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));
    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);

    auto section = VPURT::getBufferSection(val.getType().cast<vpux::NDTypeInterface>().getMemoryKind());

    auto swizzlingKey = origOp.swizzlingKeyAttr();

    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, nullptr, offset, swizzlingKey);

    return mlir::success();
}

//
// FeasibleAllocationPass
//

class FeasibleAllocationPass final : public VPUIP::FeasibleAllocationBase<FeasibleAllocationPass> {
public:
    FeasibleAllocationPass(VPUIP::MemKindCreateFunc memKindCb, VPUIP::MemKindCreateFunc secondLevelmemKindCb,
                           Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    void updateAsyncExecuteOpPosition(mlir::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
                                      llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);
    void assignCyclesToExecOps(AsyncDepsInfo& depsInfo, FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPUIP::MemKindCreateFunc _secondLvlMemKindCb;
    VPU::MemoryKind _memKind{vpux::VPU::MemoryKind::DDR};
    mlir::Optional<VPU::MemoryKind> _secondLvlMemKind{vpux::VPU::MemoryKind::DDR};
    mlir::SymbolRefAttr _memKindAttr;
    bool _enableScheduleStatistics{false};
};

FeasibleAllocationPass::FeasibleAllocationPass(VPUIP::MemKindCreateFunc memKindCb,
                                               VPUIP::MemKindCreateFunc secondLvlmemKindCb, Logger log)
        : _memKindCb(std::move(memKindCb)), _secondLvlMemKindCb(std::move(secondLvlmemKindCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult FeasibleAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.hasValue()) {
        return mlir::failure();
    }

    _memKind = maybeMemKind.getValue();
    _memKindAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(_memKind));

    _secondLvlMemKind = (_secondLvlMemKindCb != nullptr ? _secondLvlMemKindCb(secondLvlMemSpaceName.getValue()) : None);

    return mlir::success();
}

// This method will update all AsyncExecOp position in the block so that their
// order is aligned with order generated by list-scheduler. All operations will
// appear in non-descending order of start time. Such reordering is needed as
// execution order has more constraints than topological order that IR is
// aligned with. Without such sorting insertion of token dependency might hit
// an error.
void FeasibleAllocationPass::updateAsyncExecuteOpPosition(
        mlir::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
        llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps) {
    // Update placement of AsyncExecuteOps
    mlir::Operation* prevAsyncOp = nullptr;
    for (auto& schedOp : scheduledOps) {
        if (!schedOp.isOriginalOp()) {
            continue;
        }
        mlir::Operation* asyncOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        VPUX_THROW_UNLESS(asyncOp != nullptr, "AsyncOp not located based on index");
        if (prevAsyncOp != nullptr) {
            asyncOp->moveAfter(prevAsyncOp);
        } else {
            // For the first element place it before current first async exec op
            auto firstAsyncExecOp = *(netFunc.getOps<mlir::async::ExecuteOp>().begin());
            asyncOp->moveBefore(firstAsyncExecOp);
        }
        prevAsyncOp = asyncOp;
    }
}

// This method will check cycles after spilling and remove stall regions which
// might have been introduced by spilling optimizations and assign the cycle start
// and cycle end attribute to the async.execute operation
void FeasibleAllocationPass::assignCyclesToExecOps(AsyncDepsInfo& depsInfo,
                                                   FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps) {
    // find stalls on all pipelines
    auto stallsToRemove = getStallsOnAllExecutorPipelines(scheduledOps);

    // remove stalls from operations
    for (auto& schedOp : scheduledOps) {
        // sum stalls to current cycle
        size_t stallSize = 0;
        auto stalls = stallsToRemove.begin();
        while (stalls != stallsToRemove.end() && stalls->first < schedOp.cycleBegin_) {
            stallSize += checked_cast<size_t>(stalls->second - stalls->first);
            ++stalls;
        }
        // update cycles
        schedOp.cycleBegin_ -= stallSize;
        schedOp.cycleEnd_ -= stallSize;
        // store cycles for barrier scheduler
        auto execOp = depsInfo.getExecuteOpAtIndex(schedOp.op_);
        execOp->setAttr(cycleBegin, getIntAttr(execOp->getContext(), schedOp.cycleBegin_));
        execOp->setAttr(cycleEnd, getIntAttr(execOp->getContext(), schedOp.cycleEnd_));
        if (schedOp.executor == VPU::ExecutorKind::DMA_NN) {
            SmallVector<uint64_t> executorInstanceMaskVec;
            for (auto portIdx : schedOp.executorInstanceMask.set_bits()) {
                executorInstanceMaskVec.push_back(portIdx);
            }
            VPUIP::VPUIPDialect::setExecutorInstanceMask(
                    execOp, getIntArrayAttr(execOp->getContext(), executorInstanceMaskVec));
        }
    }
}

void FeasibleAllocationPass::safeRunOnModule() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    parseEnv("IE_VPUX_ENABLE_SCHEDULE_STATISTICS", _enableScheduleStatistics);

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    // cluster information
    auto nceCluster = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");
    auto nceClusterCount = nceCluster.count();

    auto dmaPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    auto dmaCount = dmaPorts.count();

    // linear scan
    auto available = IE::getAvailableMemory(module, _memKindAttr);
    const auto maxSize = available.size();
    uint64_t alignment = vpux::DEFAULT_CMX_ALIGNMENT;

    SmallVector<std::pair<vpux::AddressType, vpux::AddressType>> reservedMemVec;

    // Check for profiling reserved memory:
    if (auto dmaProfMem = IE::getDmaProfilingReservedMemory(module, _memKindAttr)) {
        auto dmaProfMemSize = dmaProfMem.byteSize();
        VPUX_THROW_UNLESS(dmaProfMem.offset().hasValue(), "No offset setting provided");
        auto dmaProfMemOffset = dmaProfMem.offset().getValue();
        VPUX_THROW_UNLESS(dmaProfMemOffset + dmaProfMemSize <= maxSize.count(),
                          "Reserved DMA profiling memory beyond available memory");

        reservedMemVec.push_back(std::make_pair(dmaProfMemOffset, dmaProfMemSize));
        _log.trace("DMA profiling reserved memory - offset: '{0}', size: '{1}'", dmaProfMemOffset, dmaProfMemSize);
    }

    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), reservedMemVec, alignment);
    auto& aliasesInfo = getChildAnalysis<AliasesInfo>(netFunc);
    auto& liveRangeInfo = getChildAnalysis<MemLiveRangeInfo>(netFunc);
    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    // VPUNN cost model
    const auto arch = VPU::getArch(module);
    const auto costModel = VPU::createCostModel(arch);

    // Copy classes for iteration with prefetch edges, as for prefetching
    // scheduler will run twice and first iteration is used to gather information
    // about the schedule and second one will perform the final allocation
    auto prefetchScan = scan;
    auto prefetchLiveRangeInfo = liveRangeInfo;

    // If schedule analysis is enabled dynamic spilling stats will be gathered
    vpux::SpillStats dynamicSpillingBeforePrefetching;
    vpux::SpillStats dynamicSpillingAfterPrefetching;
    vpux::SpillStats dynamicSpillingAfterSpillOptimizations;

    // feasible memory scheduler - list scheduler
    FeasibleMemoryScheduler scheduler(_memKind, liveRangeInfo, depsInfo, aliasesInfo, _log, scan, arch, costModel,
                                      nceClusterCount, dmaCount, _enableScheduleStatistics);

    // 1. initial schedule
    auto scheduledOps = scheduler.generateSchedule();

    if (_enableScheduleStatistics) {
        dynamicSpillingBeforePrefetching = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 2. prefetching
    FeasibleMemoryScheduler::scheduleWithPrefetch prefetchSchedule;
    OverlapDMAandDPU optimalOverlap(scheduledOps, depsInfo);
    optimalOverlap.generatePrefetchEdgesFromOverlap(prefetchSchedule);
    if (!prefetchSchedule.empty()) {
        FeasibleMemoryScheduler schedulerWithPrefetch(_memKind, prefetchLiveRangeInfo, depsInfo, aliasesInfo, _log,
                                                      prefetchScan, arch, costModel, nceClusterCount, dmaCount,
                                                      _enableScheduleStatistics);
        scheduledOps = schedulerWithPrefetch.generateSchedule(prefetchSchedule);
        scan = prefetchScan;
    }
    // TODO: recurse to strategy with useful info

    if (_enableScheduleStatistics) {
        dynamicSpillingAfterPrefetching = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 3. optimize spills
    // Locate first async-exec-op that will be used to determine insertion point for
    // new allocation operations
    auto allocOpInsertionPoint = depsInfo.getExecuteOpAtIndex(scheduledOps.begin()->op_).getOperation();
    VPUX_THROW_UNLESS(allocOpInsertionPoint != nullptr, "Unable to find insertion point for new allocation operations");
    FeasibleMemorySchedulerSpilling spilling(allocOpInsertionPoint, _memKind, _secondLvlMemKind, depsInfo, aliasesInfo,
                                             _log, scan);
    spilling.optimizeDataOpsSpills(scheduledOps);
    spilling.removeComputeOpRelocationSpills(scheduledOps);
    spilling.removeRedundantSpillWrites(scheduledOps);

    if (_enableScheduleStatistics) {
        dynamicSpillingAfterSpillOptimizations = vpux::getDynamicSpillingStats(scheduledOps);
    }

    // 3. re-order the IR
    updateAsyncExecuteOpPosition(netFunc, depsInfo, scheduledOps);

    // 4. insert spill dmas
    spilling.insertSpillCopyOps(scheduledOps);

    // 5. add cycle info to async.execute
    assignCyclesToExecOps(depsInfo, scheduledOps);

    // 6. update dependencies
    // Recreate aliasesInfo after spill insertion to get updated information about
    // root buffers of affected spill result users.
    aliasesInfo = AliasesInfo{netFunc};
    FeasibleMemorySchedulerControlEdges controlEdges(_memKind, depsInfo, aliasesInfo, _log, scan);
    // controlEdges.insertDependenciesBasic(scheduledOps); // Old method, maintained only for debug
    controlEdges.insertMemoryControlEdges(scheduledOps);
    // Linearize DMA tasks before unrolling will introduce additional dependency across different DMA engines.
    // But it's fine for single DMA engine. So insert depenency to simplify barrier scheduling.
    if (dmaCount == 1) {
        controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::DMA_NN);
    }
    // The execution for other executors is linear - based on cycles so dependencies can be inserted
    // to simplify barrier scheduling
    controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::NCE);
    controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::SHAVE_UPA);
    controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::SHAVE_ACT);

    controlEdges.updateDependenciesInIR();

    if (_enableScheduleStatistics) {
        // verify all dependencies preserved for correct analysis
        verifyDependenciesPreservedInCycles(depsInfo, scheduledOps);

        // schedule statistics
        printScheduleStatistics(netFunc, depsInfo, _log, scheduledOps);

        // dynamic spilling statistics
        printSpillingStatistics(_log, dynamicSpillingBeforePrefetching, dynamicSpillingAfterPrefetching,
                                dynamicSpillingAfterSpillOptimizations);

        // create a tracing JSON
        createTracingJSON(netFunc);
    }

    // 7. convert to allocated ops
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::Alloc>([&](VPURT::Alloc op) {
        const auto type = op.buffer().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::AllocDistributed>([&](VPURT::AllocDistributed op) {
        const auto type = op.buffer().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemRefAllocRewrite>(scan.handler(), &ctx, _log);
    patterns.add<AllocRewrite>(scan.handler(), &ctx, _log);
    patterns.add<AllocDistributedRewrite>(scan.handler(), &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
        return;
    }

    IE::setUsedMemory(module, _memKindAttr, scan.handler().maxAllocatedSize());
}

}  // namespace

//
// createFeasibleAllocationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFeasibleAllocationPass(MemKindCreateFunc memKindCb,
                                                                      MemKindCreateFunc secondLvlmemKindCb,
                                                                      Logger log) {
    return std::make_unique<FeasibleAllocationPass>(std::move(memKindCb), std::move(secondLvlmemKindCb), log);
}
