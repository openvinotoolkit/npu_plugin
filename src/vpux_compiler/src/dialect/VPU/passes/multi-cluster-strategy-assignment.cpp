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

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// GenericNCEtoNCEClusterTiling
//
template <class ConcreteOp>
class GenericNCEtoNCEClusterTiling final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericNCEtoNCEClusterTiling(mlir::MLIRContext* ctx, const StrategyManager& strategyManager, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _strategyManager(strategyManager), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const StrategyManager& _strategyManager;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericNCEtoNCEClusterTiling<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got operation {0}", origOp);

    if (origOp->template getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }
    auto activationTensorDistributionMode = _strategyManager.getActivationTensorDistributionMode(origOp);
    auto weightsTensorDistributionMode = _strategyManager.getWeightsTensorDistributionMode(origOp);
    auto activationTensorNumTiles = _strategyManager.getActivationTensorNumTiles(origOp);
    auto weightTensorNumTiles = _strategyManager.getWeightsTensorNumTiles(origOp);

    auto distributedActivationCopyOp = _strategyManager.createDistributedInputTensor(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedWeightsCopyOp = _strategyManager.createDistributedInputTensor(
            origOp, origOp.filter(), weightsTensorDistributionMode, weightTensorNumTiles);

    auto distributedOutputTensorType = _strategyManager.createDistributedOutputTensorType(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().template cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    auto clusterTilingOp = rewriter.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightsCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<VPU::YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<VPU::NCEClusterTilingOp>(
            clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

template <>
mlir::LogicalResult GenericNCEtoNCEClusterTiling<VPU::NCEMaxPoolOp>::matchAndRewrite(
        VPU::NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got operation {0}", origOp);

    if (origOp->template getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    const llvm::StringRef strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();

    auto activationTensorDistributionMode = _strategyManager.getActivationTensorDistributionMode(origOp);
    auto activationTensorNumTiles = _strategyManager.getActivationTensorNumTiles(origOp);

    auto distributedActivationCopyOp = _strategyManager.createDistributedInputTensor(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedOutputTensorType = _strategyManager.createDistributedOutputTensorType(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    auto clusterTilingOp = rewriter.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType, mlir::ValueRange{distributedActivationCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<VPU::YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<VPU::NCEClusterTilingOp>(
            clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

template <>
mlir::LogicalResult GenericNCEtoNCEClusterTiling<VPU::NCEEltwiseOp>::matchAndRewrite(
        VPU::NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got operation {0}", origOp);

    if (origOp->template getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto activationTensorDistributionMode = _strategyManager.getActivationTensorDistributionMode(origOp);
    auto activationTensorNumTiles = _strategyManager.getActivationTensorNumTiles(origOp);

    auto distributedActivationCopyOp1 = _strategyManager.createDistributedInputTensor(
            origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedActivationCopyOp2 = _strategyManager.createDistributedInputTensor(
            origOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedOutputTensorType = _strategyManager.createDistributedOutputTensorType(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    auto clusterTilingOp = rewriter.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp1->getResult(0), distributedActivationCopyOp2->getResult(0)},
            bodyBuilder);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<VPU::YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<VPU::NCEClusterTilingOp>(
            clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// MultiClusterStrategyAssignmentPass
//

class MultiClusterStrategyAssignmentPass final :
        public VPU::MultiClusterStrategyAssignmentBase<MultiClusterStrategyAssignmentPass> {
public:
    explicit MultiClusterStrategyAssignmentPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MultiClusterStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& ctx = getContext();
    StrategyManager strategyManager(func, _log, &ctx);
    strategyManager.computeOptimalMultiClusterStrategy();
    // strategyManager.removeStrategyAttribute();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<GenericNCEtoNCEClusterTiling<VPU::NCEConvolutionOp>>(&ctx, strategyManager, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<VPU::NCEDepthConvolutionOp>>(&ctx, strategyManager, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<VPU::NCEMaxPoolOp>>(&ctx, strategyManager, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<VPU::NCEEltwiseOp>>(&ctx, strategyManager, _log);

    mlir::ConversionTarget target(ctx);

    // If an operation does not have multi-cluster strategy, it doesn't fit in CMX, it will be tiled instead.
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (op->hasAttr(multiClusterStrategy)) {
            return (op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr);
        } else
            return true;
    });

    target.addLegalOp<VPU::NCEClusterTilingOp>();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMultiClusterStrategyAssignmentPass
//

std::unique_ptr<mlir::Pass> VPU::createMultiClusterStrategyAssignmentPass(Logger log) {
    return std::make_unique<MultiClusterStrategyAssignmentPass>(log);
}