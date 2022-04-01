//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/core/prefetch_edge_generator.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

//
// AllocRewrite
//

class AllocRewrite final : public mlir::OpRewritePattern<mlir::memref::AllocOp> {
public:
    AllocRewrite(LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::AllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    LinearScanHandler& _allocInfo;
    Logger _log;
};

mlir::LogicalResult AllocRewrite::matchAndRewrite(mlir::memref::AllocOp origOp, mlir::PatternRewriter& rewriter) const {
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
    rewriter.replaceOpWithNewOp<IERT::StaticAllocOp>(origOp, val.getType(), offset);

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

    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, val.getType(), section, offset);

    return mlir::success();
}

//
// FeasibleAllocationPass
//

class FeasibleAllocationPass final : public IERT::FeasibleAllocationBase<FeasibleAllocationPass> {
public:
    FeasibleAllocationPass(IERT::MemKindCreateFunc memKindCb, IERT::MemKindCreateFunc secondLevelmemKindCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    void updateAsyncExecuteOpPosition(mlir::FuncOp& netFunc, AsyncDepsInfo& depsInfo,
                                      llvm::ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

private:
    IERT::MemKindCreateFunc _memKindCb;
    IERT::MemKindCreateFunc _secondLvlMemKindCb;
    VPU::MemoryKind _memKind;
    mlir::Optional<VPU::MemoryKind> _secondLvlMemKind;
    mlir::SymbolRefAttr _memKindAttr;
};

FeasibleAllocationPass::FeasibleAllocationPass(IERT::MemKindCreateFunc memKindCb,
                                               IERT::MemKindCreateFunc secondLvlmemKindCb, Logger log)
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

void FeasibleAllocationPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    // linear scan
    auto available = IE::getAvailableMemory(module, _memKindAttr);
    const auto maxSize = available.size();
    const uint64_t alignment = 64;

    LinearScan<mlir::Value, LinearScanHandler> scan(maxSize.count(), alignment);
    auto& aliasesInfo = getChildAnalysis<AliasesInfo>(netFunc);
    auto& liveRangeInfo = getChildAnalysis<MemLiveRangeInfo>(netFunc);
    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    // Copy classes for iteration with prefetch edges, as for prefetching
    // scheduler will run twice and first iteration is used to gather information
    // about the schedule and second one will perform the final allocation
    auto prefetchScan = scan;
    auto prefetchLiveRangeInfo = liveRangeInfo;

    // feasible memory scheduler - list scheduler
    FeasibleMemoryScheduler scheduler(_memKind, liveRangeInfo, depsInfo, aliasesInfo, _log, scan);

    // 1. initial schedule
    auto scheduledOps = scheduler.generateSchedule();

    // 2. prefetching
    // 2.1. optimization for initial schedule - generating prefetch edges
    PrefetchEdgeGenerator PrefetchEdgeGenerator(scheduledOps, depsInfo);
    auto prefetchEdges = PrefetchEdgeGenerator.generatePrefetchEdges();

    // 2.2. schedule again with prefetching
    if (!prefetchEdges.empty()) {
        FeasibleMemoryScheduler schedulerWithPrefetch(_memKind, prefetchLiveRangeInfo, depsInfo, aliasesInfo, _log,
                                                      prefetchScan);
        scheduledOps = schedulerWithPrefetch.generateSchedule(prefetchEdges);
        scan = prefetchScan;
    }

    // 3. optimize spills
    FeasibleMemorySchedulerSpilling spilling(netFunc, _memKind, _secondLvlMemKind, depsInfo, aliasesInfo, _log, scan);
    spilling.optimizeDataOpsSpills(scheduledOps);
    spilling.removeComputeOpRelocationSpills(scheduledOps);
    spilling.removeRedundantSpillWrites(scheduledOps);

    // 3. re-order the IR
    updateAsyncExecuteOpPosition(netFunc, depsInfo, scheduledOps);

    // 4. insert spill dmas
    spilling.insertSpillCopyOps(scheduledOps);

    // 5. update dependencies
    // Recreate aliasesInfo after spill insertion to get updated information about
    // root buffers of affected spill result users.
    aliasesInfo = AliasesInfo{netFunc};
    FeasibleMemorySchedulerControlEdges controlEdges(_memKind, depsInfo, aliasesInfo, _log, scan);
    // controlEdges.insertDependenciesBasic(scheduledOps); // Old method, maintained only for debug
    controlEdges.insertMemoryControlEdges(scheduledOps);
    // Insert dependencies aligned with schedule order to properly serialize
    // operations on a given executor. This is needed for operations which might not have
    // any input data flow like dataOps (data DMAs)
    controlEdges.insertScheduleOrderDepsForExecutor(scheduledOps, VPU::ExecutorKind::DMA_NN);

    controlEdges.updateDependenciesInIR();

    // 6. convert to allocated ops
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::AllocDistributed>([&](VPURT::AllocDistributed op) {
        const auto type = op.buffer().getType().cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });

    mlir::RewritePatternSet patterns(&ctx);
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

std::unique_ptr<mlir::Pass> vpux::IERT::createFeasibleAllocationPass(MemKindCreateFunc memKindCb,
                                                                     MemKindCreateFunc secondLvlmemKindCb, Logger log) {
    return std::make_unique<FeasibleAllocationPass>(std::move(memKindCb), std::move(secondLvlmemKindCb), log);
}
