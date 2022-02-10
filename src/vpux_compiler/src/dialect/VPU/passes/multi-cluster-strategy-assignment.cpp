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
// ConvToMultiCluster
//

class ConvToMultiCluster final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    ConvToMultiCluster(mlir::MLIRContext* ctx, const StrategyManager& strategyManager, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _strategyManager(strategyManager), _log(log) {
        setDebugName("ConvToMultiCluster");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const StrategyManager& _strategyManager;
    Logger _log;
};

mlir::LogicalResult ConvToMultiCluster::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    vpux::VPU::DistributionMode activationTensorDistributionMode;
    vpux::VPU::DistributionMode weightsTensorDistributionMode;
    mlir::ArrayAttr activationTensorNumTiles;
    mlir::ArrayAttr weightTensorNumTiles;

    _log.trace("Got operation {0}", origOp);

    if (origOp->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped into NCEClusterTiling");
    }

    // Retrieve the strategy
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();

    if (strategy == splitOverHeightOverLappedStrategy) {
        activationTensorDistributionMode = vpux::VPU::DistributionMode::overlapped;
        weightsTensorDistributionMode = vpux::VPU::DistributionMode::multicasted;
        activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 4, 1}));
        weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == splitOverHeightStrategy) {
        activationTensorDistributionMode = vpux::VPU::DistributionMode::overlapped;
        weightsTensorDistributionMode = vpux::VPU::DistributionMode::multicasted;
        activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 4, 1}));
        weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == splitOverKernelStrategy) {
        activationTensorDistributionMode = vpux::VPU::DistributionMode::multicasted;
        weightsTensorDistributionMode = vpux::VPU::DistributionMode::segmented;
        activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
        weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 4, 1}));
    } else {
        VPUX_THROW("Operation '{0}' does not have a valid multi-cluster strategy", origOp);
    }

    // Create the copy ops for the distributed activation tensor
    auto distributedActivationCopyOp = _strategyManager.createDistributedActivationTensor(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    // Create the copy ops for the distributed weights tensor
    auto distributedWeightsCopyOp = _strategyManager.createDistributedWeightsTensor(
            origOp, weightsTensorDistributionMode, weightTensorNumTiles);

    // Create the copy ops for the distributed output tensor type
    auto distributedOutputTensorType = _strategyManager.createDistributedOutputTensorType(
            origOp, activationTensorDistributionMode, activationTensorNumTiles);

    // Set the output of the VPU::NCEConvolutionOp to be in CMX
    auto origOutput = origOp->getResult(0);
    const auto cmxMemSpace =
            changeMemSpace(origOutput.getType().cast<mlir::RankedTensorType>(), VPU::MemoryKind::CMX_NN);
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
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0]);
        builder.create<VPU::YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp =
            rewriter.create<VPU::NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOp.output().getType(),
                                                     clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    rewriter.replaceOp(origOp, outputCopyOp->getResults());
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
    patterns.insert<ConvToMultiCluster>(&ctx, strategyManager, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto nceConvOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
            return (op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr);
        }
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

std::unique_ptr<mlir::Pass> vpux::VPU::createMultiClusterStrategyAssignmentPass(Logger log) {
    return std::make_unique<MultiClusterStrategyAssignmentPass>(log);
}