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
// MaxPoolToNCEClusterTiling
//

class MaxPoolToNCEClusterTiling final : public mlir::OpRewritePattern<VPU::NCEMaxPoolOp> {
public:
    MaxPoolToNCEClusterTiling(mlir::MLIRContext* ctx, const StrategyManager& strategyManager, Logger log)
            : mlir::OpRewritePattern<VPU::NCEMaxPoolOp>(ctx), _strategyManager(strategyManager), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const StrategyManager& _strategyManager;
    Logger _log;
};

mlir::LogicalResult MaxPoolToNCEClusterTiling::matchAndRewrite(VPU::NCEMaxPoolOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got operation {0}", origOp);

    if (origOp->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped into NCEClusterTiling");
    }

    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();

    auto activationTensorDistributionMode = StrategyManager::getActivationTensorDistributionMode(strategy);
    auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), StrategyManager::getActivationTensorNumTiles(strategy));

    // Create the copy ops for the distributed activation tensor
    auto distributedActivationCopyOp = _strategyManager.createDistributedActivationTensorforMaxPool(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    // Create the copy ops for the distributed output tensor type
    auto distributedOutputTensorType = _strategyManager.createDistributedOutputTensorTypeforMaxPool(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    // // save original output memspace
    const auto origOutType = origOp.output().getType();
    const auto origOutMemSpace = IE::getMemorySpace(origOutType.cast<mlir::RankedTensorType>());
    Logger::global().error("origOutMemSpace original {0}", origOutMemSpace);

    // Set the output of the VPU::NCEConvolutionOp to be in CMX
    auto origOutput = origOp->getResult(0);
    const auto cmxMemSpace =
            changeMemSpace(origOutput.getType().cast<mlir::RankedTensorType>(), VPU::MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);
    Logger::global().error("origOutMemSpace updated {0}", origOutput.getType());

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

    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();

    auto activationTensorDistributionMode = StrategyManager::getActivationTensorDistributionMode(strategy);
    auto weightsTensorDistributionMode = StrategyManager::getWeightsTensorDistributionMode(strategy);
    auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), StrategyManager::getActivationTensorNumTiles(strategy));
    auto weightTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), StrategyManager::getWeightsTensorNumTiles(strategy));

    // Create the copy ops for the distributed activation tensor
    auto distributedActivationCopyOp = _strategyManager.createDistributedActivationTensor(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    // Create the copy ops for the distributed weights tensor
    auto distributedWeightsCopyOp = _strategyManager.createDistributedWeightsTensor(
            origOp, weightsTensorDistributionMode, weightTensorNumTiles);

    // Create the copy ops for the distributed output tensor type
    auto distributedOutputTensorType = _strategyManager.createDistributedOutputTensorType(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    // save original output memspace
    const auto origOutType = origOp.output().getType();
    const auto origOutMemSpace = IE::getMemorySpace(origOutType.template cast<mlir::RankedTensorType>());
    Logger::global().error("origOutMemSpace original {0}", origOutMemSpace);

    // Set the output of the VPU::NCEConvolutionOp to be in CMX
    auto origOutput = origOp->getResult(0);
    const auto cmxMemSpace =
            changeMemSpace(origOutput.getType().template cast<mlir::RankedTensorType>(), VPU::MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);
    Logger::global().error("origOutMemSpace updated {0}", origOutput.getType());

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
    StrategyManager strategyManager(func, _log);
    strategyManager.computeOptimalMultiClusterStrategy();

    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<GenericNCEtoNCEClusterTiling<VPU::NCEConvolutionOp>>(&ctx, strategyManager, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<VPU::NCEDepthConvolutionOp>>(&ctx, strategyManager, _log);
    patterns.insert<MaxPoolToNCEClusterTiling>(&ctx, strategyManager, _log);

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