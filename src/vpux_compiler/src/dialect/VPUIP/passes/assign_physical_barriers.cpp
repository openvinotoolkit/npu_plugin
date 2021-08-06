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

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

constexpr int64_t MAX_BARRIERS_PER_INFERENCE = 32;
constexpr int64_t BARRIERS_PER_CLUSTER = 8;

//
// BarrierAllocation
//

class BarrierAllocation final {
public:
    BarrierAllocation(mlir::Operation* op, int64_t numBarriers, Logger log);

    int64_t getID(mlir::Value val) const;

private:
    llvm::DenseMap<mlir::Value, int64_t> _idMap;
};

// TODO: [#6150] Implement safe static barriers allocation
BarrierAllocation::BarrierAllocation(mlir::Operation* op, int64_t numBarriers, Logger log) {
    int64_t barrierID = 0;

    log.trace("Assign {0} physical barriers", numBarriers);

    const auto callback = [&](VPUIP::DeclareVirtualBarrierOp virtOp) {
        _idMap.insert({virtOp.barrier(), barrierID});
        barrierID = (barrierID + 1) % numBarriers;
    };

    op->walk(callback);
}

int64_t BarrierAllocation::getID(mlir::Value val) const {
    const auto it = _idMap.find(val);
    VPUX_THROW_UNLESS(it != _idMap.end(), "Value '{0}' was not covered by BarrierAllocation");
    return it->second;
}

//
// VirtualBarrierRewrite
//

class VirtualBarrierRewrite final : public mlir::OpRewritePattern<VPUIP::DeclareVirtualBarrierOp> {
public:
    VirtualBarrierRewrite(mlir::MLIRContext* ctx, const BarrierAllocation& allocInfo, Logger log)
            : mlir::OpRewritePattern<VPUIP::DeclareVirtualBarrierOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::DeclareVirtualBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const BarrierAllocation& _allocInfo;
    Logger _log;
};

mlir::LogicalResult VirtualBarrierRewrite::matchAndRewrite(VPUIP::DeclareVirtualBarrierOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DeclareVirtualBarrierOp Operation '{0}'", origOp->getLoc());

    const auto barrierID = _allocInfo.getID(origOp.barrier());
    _log.nest().trace("Use physical barrier ID '{0}'", barrierID);

    rewriter.replaceOpWithNewOp<VPUIP::ConfigureBarrierOp>(origOp, barrierID);
    return mlir::success();
}

//
// AssignPhysicalBarriersPass
//

class AssignPhysicalBarriersPass final : public VPUIP::AssignPhysicalBarriersBase<AssignPhysicalBarriersPass> {
public:
    explicit AssignPhysicalBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignPhysicalBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto nceAttr = VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_Cluster);
    auto nceResOp = resOp.getExecutor(nceAttr);
    VPUX_THROW_UNLESS(nceResOp != nullptr, "Failed to get NCE_Cluster information");

    const auto numClusters = nceResOp.count();
    const auto maxNumBarriers = std::min(MAX_BARRIERS_PER_INFERENCE, BARRIERS_PER_CLUSTER * numClusters);

    const auto numBarriers = _numBarriersOpt.hasValue() ? _numBarriersOpt.getValue() : maxNumBarriers;
    VPUX_THROW_UNLESS(numBarriers > 0 && numBarriers <= maxNumBarriers,
                      "Number of physical barriers '{0}' is out of range '{1}'", numBarriers, maxNumBarriers);

    BarrierAllocation allocInfo(func, numBarriers, _log);

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPUIP::DeclareVirtualBarrierOp>();
    target.addLegalOp<VPUIP::ConfigureBarrierOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<VirtualBarrierRewrite>(&ctx, allocInfo, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAssignPhysicalBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAssignPhysicalBarriersPass(Logger log) {
    return std::make_unique<AssignPhysicalBarriersPass>(log);
}
