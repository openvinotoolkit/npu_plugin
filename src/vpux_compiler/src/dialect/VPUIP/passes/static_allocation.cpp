//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseSet.h>

using namespace vpux;

namespace {

using LinearScanImpl = LinearScan<mlir::Value, LinearScanHandler>;

//
// AllocRewrite
//

template <class ConcreteAllocOp>
class AllocRewrite final : public mlir::OpRewritePattern<ConcreteAllocOp> {
public:
    AllocRewrite(const LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteAllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteAllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const LinearScanHandler& _allocInfo;
    Logger _log;
};

template <class ConcreteAllocOp>
void createAllocOp(mlir::PatternRewriter&, ConcreteAllocOp, mlir::Type, int64_t) {
    VPUX_THROW("Unsupported allocation operation type");
}

template <>
void createAllocOp(mlir::PatternRewriter& rewriter, mlir::memref::AllocOp origOp, mlir::Type type, int64_t offset) {
    rewriter.replaceOpWithNewOp<VPUIP::StaticAllocOp>(origOp, type, offset);
}

template <>
void createAllocOp(mlir::PatternRewriter& rewriter, VPURT::Alloc origOp, mlir::Type type, int64_t offset) {
    auto section = VPURT::getBufferSection(type.cast<vpux::NDTypeInterface>().getMemoryKind());
    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, type, section, nullptr, offset,
                                                        origOp.getSwizzlingKeyAttr());
}

template <class ConcreteAllocOp>
mlir::LogicalResult AllocRewrite<ConcreteAllocOp>::matchAndRewrite(ConcreteAllocOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}' - '{1}'", origOp->getName(), origOp->getLoc());

    const auto val = origOp->getResult(0);

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.template getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));

    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);
    createAllocOp(rewriter, origOp, val.getType(), offset);

    return mlir::success();
}

//
// StaticAllocationPass
//

class StaticAllocationPass final : public VPUIP::StaticAllocationBase<StaticAllocationPass> {
public:
    StaticAllocationPass(VPUIP::MemKindCreateFunc memKindCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    LinearScanHandler runLinearScan(mlir::func::FuncOp netFunc);

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPU::MemoryKind _memKind{VPU::MemoryKind::DDR};
    mlir::StringAttr _memKindAttr = nullptr;
};

StaticAllocationPass::StaticAllocationPass(VPUIP::MemKindCreateFunc memKindCb, Logger log)
        : _memKindCb(std::move(memKindCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult StaticAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.has_value()) {
        return mlir::failure();
    }

    _memKind = maybeMemKind.value();
    _memKindAttr = mlir::StringAttr::get(ctx, stringifyEnum(_memKind));

    return mlir::success();
}

LinearScanHandler StaticAllocationPass::runLinearScan(mlir::func::FuncOp netFunc) {
    auto& aliasInfo = getChildAnalysis<AliasesInfo>(netFunc);
    auto& liveRangeInfo = getChildAnalysis<MemLiveRangeInfo>(netFunc);
    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(netFunc);

    auto module = netFunc->getParentOfType<mlir::ModuleOp>();
    auto availableMem = IE::getAvailableMemory(module, _memKindAttr);
    VPUX_THROW_WHEN(availableMem == nullptr, "The memory space '{0}' is not available", _memKind);

    const Byte maxMemSize = availableMem.size();
    const uint64_t memDefaultAlignment = 64;  // TODO: extract from run-time resources information?

    LinearScanImpl scan(maxMemSize.count(), {}, memDefaultAlignment);

    const auto allocNewBuffers = [&](const ValueOrderedSet& usedBufs) {
        _log.trace("Locate new buffers");
        _log = _log.nest();

        SmallVector<mlir::Value> newBufs;

        for (auto val : usedBufs) {
            const auto type = val.getType().cast<vpux::NDTypeInterface>();
            if (type.getMemoryKind() != _memKind) {
                continue;
            }

            _log.trace("Check buffer '{0}'", val);

            if (scan.handler().isAlive(val)) {
                continue;
            }

            _log.nest().trace("This task is the first usage of the buffer, allocate it");

            scan.handler().markAsAlive(val);
            newBufs.push_back(val);
        }

        _log.trace("Allocate memory for the new buffers");
        VPUX_THROW_UNLESS(scan.alloc(newBufs, /*allowSpills*/ false), "Failed to statically allocate '{0}' memory",
                          _memKind);

        _log = _log.unnest();
    };

    const auto freeDeadBuffers = [&](const ValueOrderedSet& usedBufs) {
        _log.trace("Free dead buffers");
        _log = _log.nest();

        for (auto val : usedBufs) {
            _log.trace("Mark as dead buffer '{0}'", val);
            scan.handler().markAsDead(val);
        }

        _log.trace("Free memory for the dead buffers");
        scan.freeNonAlive();

        _log = _log.unnest();
    };

    auto getFreeBuffers = [&](const ValueOrderedSet& usedBufs, mlir::async::ExecuteOp op) {
        ValueOrderedSet freeBuffers;

        _log.trace("Locate dead buffers");
        _log = _log.nest();

        for (auto val : usedBufs) {
            const auto type = val.getType().cast<vpux::NDTypeInterface>();
            if (type.getMemoryKind() != _memKind) {
                continue;
            }

            _log.trace("Check buffer '{0}'", val);

            if (liveRangeInfo.eraseUser(val, op) == 0) {
                _log.nest().trace("This bucket is the last usage of the buffer, store it");
                freeBuffers.insert(val);
            }
        }

        _log = _log.unnest();

        return freeBuffers;
    };

    // Store buffers with their end cycle
    std::map<size_t, ValueOrderedSet> freeBuffersCycleEnd;

    mlir::async::ExecuteOp prevExecOp;
    std::list<ScheduledOpOneResource> scheduledOpsResources;
    for (auto curExecOp : netFunc.getOps<mlir::async::ExecuteOp>()) {
        // retrieve async.execute execution cycles
        auto cycleBegin = getAsyncExecuteCycleBegin(curExecOp);
        auto cycleEnd = getAsyncExecuteCycleEnd(curExecOp);

        _log.trace("Process next task at '{0}' during cycles '{1}' to '{2}'", curExecOp->getLoc(), cycleBegin,
                   cycleEnd);
        _log = _log.nest();

        // Free buffers if the current async.execute operation is executing
        // after the end cycle for the buffers
        for (auto& freeBuffers : freeBuffersCycleEnd) {
            // skip entries with no buffers
            if (freeBuffers.second.empty()) {
                continue;
            }
            // check if current async.execute starts before buffer end cycle
            if (cycleBegin < freeBuffers.first) {
                continue;
            }
            _log.nest().trace("Current cycle '{0}', freeing buffers end at cycle '{1}'", cycleBegin, freeBuffers.first);
            freeDeadBuffers(freeBuffers.second);
            freeBuffers.second.clear();
        }

        const auto usedBufs = liveRangeInfo.getUsedBuffers(curExecOp);

        allocNewBuffers(usedBufs);

        auto opIndex = depsInfo.getIndex(curExecOp);

        // buffers used by operation, both inputs and outputs
        mlir::DenseSet<mlir::Value> inputBuffers;
        mlir::DenseSet<mlir::Value> outputBuffers;

        // Get operation buffers for all operands. Go through each layer op and
        // store in a set all root buffers
        auto* bodyBlock = &curExecOp.body().front();
        for (auto& innerOp : bodyBlock->getOperations()) {
            if (!mlir::isa<VPUIP::LayerOpInterface>(innerOp)) {
                continue;
            }

            auto inputs = mlir::dyn_cast<VPUIP::LayerOpInterface>(innerOp).getInputs();
            for (const auto& input : inputs) {
                const auto type = input.getType().cast<vpux::NDTypeInterface>();
                if (type == nullptr || type.getMemoryKind() != _memKind) {
                    continue;
                }
                auto rootBuffers = aliasInfo.getRoots(input);
                VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", input,
                                  rootBuffers.size());
                auto rootBuffer = *rootBuffers.begin();
                inputBuffers.insert(rootBuffer);
            }

            auto outputs = mlir::dyn_cast<VPUIP::LayerOpInterface>(innerOp).getOutputs();
            for (const auto& output : outputs) {
                const auto type = output.getType().cast<vpux::NDTypeInterface>();
                if (type == nullptr || type.getMemoryKind() != _memKind) {
                    continue;
                }
                auto rootBuffers = aliasInfo.getRoots(output);
                VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}",
                                  output, rootBuffers.size());
                auto rootBuffer = *rootBuffers.begin();
                outputBuffers.insert(rootBuffer);
            }
        }

        // For all identified buffers used by operation create separate entries with information
        // about memory ranges to properly identify range producer and consumers at a given time
        for (auto& buf : inputBuffers) {
            if (!isBufAllocOp(buf.getDefiningOp())) {
                continue;
            }
            auto addressStart = scan.handler().getAddress(buf);
            auto addressEnd = addressStart + scan.handler().getSize(buf) - 1;
            _log.trace("op = '{0}'\t input = [{1} - {2}]", opIndex, addressStart, addressEnd);
            scheduledOpsResources.push_back(ScheduledOpOneResource(opIndex, addressStart, addressEnd,
                                                                   ScheduledOpOneResource::EResRelation::CONSUMER));
        }
        for (auto& buf : outputBuffers) {
            if (!isBufAllocOp(buf.getDefiningOp())) {
                continue;
            }
            auto addressStart = scan.handler().getAddress(buf);
            auto addressEnd = addressStart + scan.handler().getSize(buf) - 1;
            _log.trace("op = '{0}'\t output = [{1} - {2}]", opIndex, addressStart, addressEnd);
            scheduledOpsResources.push_back(ScheduledOpOneResource(opIndex, addressStart, addressEnd,
                                                                   ScheduledOpOneResource::EResRelation::PRODUCER));
        }

        // Store free buffers with cycle end
        freeBuffersCycleEnd[cycleEnd] = getFreeBuffers(usedBufs, curExecOp);

        _log = _log.unnest();
    }

    // Free all remaining buffers
    _log.trace("Free remaining buffers");
    for (auto freeBuffers : freeBuffersCycleEnd) {
        if (!freeBuffers.second.empty()) {
            _log.nest().trace("Freeing buffers end at cycle '{1}'", freeBuffers.first);
            freeDeadBuffers(freeBuffers.second);
        }
    }

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator<ScheduledOpOneResource> controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    _log.trace("Insert control edges for overlapping memory resources");
    _log = _log.nest();

    // Apply dependencies from controlEdges set into depsInfo.
    updateControlEdgesInDepsInfo(depsInfo, controlEdges, _log);

    _log = _log.unnest();

    // Transfer dependencies into tokens between AsyncExecuteOps
    depsInfo.updateTokenDependencies();

    return scan.handler();
}

void StaticAllocationPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    const auto allocInfo = runLinearScan(netFunc);
    IE::setUsedMemory(module, _memKindAttr, allocInfo.maxAllocatedSize());

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.memref().getType().dyn_cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::Alloc>([&](VPURT::Alloc op) {
        const auto type = op.getBuffer().getType().dyn_cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AllocRewrite<mlir::memref::AllocOp>>(allocInfo, &ctx, _log);
    patterns.add<AllocRewrite<VPURT::Alloc>>(allocInfo, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createStaticAllocationPass(MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<StaticAllocationPass>(std::move(memKindCb), log);
}
