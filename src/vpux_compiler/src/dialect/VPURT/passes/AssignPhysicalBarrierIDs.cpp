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

#include "vpux/compiler/core/runtime_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

using namespace vpux;

namespace {

static constexpr int64_t MAX_DMA_ENGINES = 2;

//
// VirtualBarrierRewrite
//

class VirtualBarrierRewrite final : public mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp> {
public:
    VirtualBarrierRewrite(mlir::MLIRContext* ctx, const RuntimeSimulator& _simulator, Logger log)
            : mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp>(ctx), _simulator(_simulator), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const RuntimeSimulator& _simulator;
    Logger _log;
};

mlir::LogicalResult VirtualBarrierRewrite::matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DeclareVirtualBarrierOp Operation '{0}'", origOp->getLoc());

    auto barrierRealandVirtualID = _simulator.getID(origOp.getOperation());
    _log.nest().trace("Use physical barrier ID '{0}'", barrierRealandVirtualID.first);

    rewriter.replaceOpWithNewOp<VPURT::ConfigureBarrierOp>(origOp, barrierRealandVirtualID.first,
                                                           barrierRealandVirtualID.second);

    return mlir::success();
}

//
// AssignPhysicalBarrierIDsPass
//

class AssignPhysicalBarrierIDsPass final : public VPURT::AssignPhysicalBarrierIDsBase<AssignPhysicalBarrierIDsPass> {
public:
    explicit AssignPhysicalBarrierIDsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignPhysicalBarrierIDsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto dmaAttr = VPU::ExecutorKindAttr::get(&ctx, VPU::ExecutorKind::DMA_NN);
    auto dmaResOp = resOp.getExecutor(dmaAttr);
    VPUX_THROW_UNLESS(dmaResOp != nullptr, "Failed to get DMA_NN information");

    const auto numDmaEngines = dmaResOp.count();
    VPUX_THROW_UNLESS(numDmaEngines <= MAX_DMA_ENGINES, "Found {0} DMA engines (max {1})", numDmaEngines,
                      MAX_DMA_ENGINES);

    RuntimeSimulator simulator(&ctx, func, _log, numDmaEngines, 8);
    simulator.assignPhysicalIDs();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPURT::DeclareVirtualBarrierOp>();
    target.addLegalOp<VPURT::ConfigureBarrierOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<VirtualBarrierRewrite>(&ctx, simulator, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAssignPhysicalBarrierIDsPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignPhysicalBarrierIDsPass(Logger log) {
    return std::make_unique<AssignPhysicalBarrierIDsPass>(log);
}
