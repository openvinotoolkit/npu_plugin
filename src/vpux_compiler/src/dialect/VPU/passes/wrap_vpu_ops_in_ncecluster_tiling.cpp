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

#include "vpux/compiler/dialect/VPU/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/loop.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// NCEConvolutionRewriterRewrite
//

class NCEConvolutionRewriter final : public mlir::OpRewritePattern<NCEConvolutionOp> {
public:
    NCEConvolutionRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<NCEConvolutionOp>(ctx), _numClusters(numClusters), _log(log) {
        setDebugName("NCEConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult NCEConvolutionRewriter::matchAndRewrite(NCEConvolutionOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    NCEClusterTilingOp clusterTilingOp = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTableTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));

    const auto arch = VPU::getArch(origOp.getOperation());
    const auto canUseCMajor =
            VPU::NCEInvariant::isChannelMajorCompatible(arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

    if (!canUseCMajor) {
        const auto activationAlignment = getActivationTensorAlignment(origOp.getOperation(), origOp.input(), strategy);
        if (activationAlignment.hasValue()) {
            activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.getValue());
        }
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(origOp.getContext(), weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(origOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(origOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles,
                                        activationAlignmentAttr, strategy);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    if (canUseCMajor) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles =
                getIntArrayAttr(origOp.getContext(), getActivationWindowTensorNumTiles(strategy));
        const auto distributedActivationWindowCopyOp =
                createDistributedCopyIn(origOp, origOp.activationWindow(), activationWindowDistributionMode,
                                        activationWindowNumTiles, nullptr, strategy);

        clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
                origOp->getLoc(), distributedOutputTensorType,
                mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightsCopyOp->getResult(0),
                                 distributedWeightTableCopyOp->getResult(0),
                                 distributedActivationWindowCopyOp->getResult(0)},
                bodyBuilder);
    } else {
        clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
                origOp->getLoc(), distributedOutputTensorType,
                mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightsCopyOp->getResult(0),
                                 distributedWeightTableCopyOp->getResult(0)},
                bodyBuilder);
    }

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEDepthConvolutionRewriterRewrite
//

class NCEDepthConvolutionRewriter final : public mlir::OpRewritePattern<NCEDepthConvolutionOp> {
public:
    NCEDepthConvolutionRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<NCEDepthConvolutionOp>(ctx), _numClusters(numClusters), _log(log) {
        setDebugName("NCEDepthConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEDepthConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult NCEDepthConvolutionRewriter::matchAndRewrite(NCEDepthConvolutionOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTableTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationWindowTensorNumTiles(strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(origOp.getContext(), weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(origOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(origOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(origOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy);

    const auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles,
                                        activationAlignmentAttr, strategy);

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightsCopyOp->getResult(0),
                             distributedWeightTableCopyOp->getResult(0),
                             distributedActivationWindowCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEMaxPoolRewriterRewrite
//

class NCEMaxPoolRewriter final : public mlir::OpRewritePattern<NCEMaxPoolOp> {
public:
    NCEMaxPoolRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<NCEMaxPoolOp>(ctx), _numClusters(numClusters), _log(log) {
        setDebugName("NCEMaxPoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult NCEMaxPoolRewriter::matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTableTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationWindowTensorNumTiles(strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(origOp.getContext(), weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(origOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(origOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy);

    const auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles,
                                        activationAlignmentAttr, strategy);

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightTableCopyOp->getResult(0),
                             distributedActivationWindowCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEEltwiseRewriterRewrite
//

class NCEEltwiseRewriter final : public mlir::OpRewritePattern<NCEEltwiseOp> {
public:
    NCEEltwiseRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<NCEEltwiseOp>(ctx), _numClusters(numClusters), _log(log) {
        setDebugName("NCEEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult NCEEltwiseRewriter::matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));

    const auto distributedActivationCopyOp1 =
            createDistributedCopyIn(origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedActivationCopyOp2 =
            createDistributedCopyIn(origOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles,
                                        activationAlignmentAttr, strategy);

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp1->getResult(0), distributedActivationCopyOp2->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// WrapVPUOpsInNCEClusterTilingPass
//

class WrapVPUOpsInNCEClusterTilingPass final :
        public WrapVPUOpsInNCEClusterTilingBase<WrapVPUOpsInNCEClusterTilingPass> {
public:
    explicit WrapVPUOpsInNCEClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void WrapVPUOpsInNCEClusterTilingPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& ctx = getContext();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClusters = nceOp.count();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<NCEConvolutionRewriter>(&ctx, numClusters, _log);
    patterns.insert<NCEDepthConvolutionRewriter>(&ctx, numClusters, _log);
    patterns.insert<NCEMaxPoolRewriter>(&ctx, numClusters, _log);
    patterns.insert<NCEEltwiseRewriter>(&ctx, numClusters, _log);

    mlir::ConversionTarget target(ctx);

    // If an operation does not have multi-cluster strategy, it doesn't fit in CMX, it will be tiled instead.
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (op->hasAttr(multiClusterStrategy)) {
            return (op->getParentOfType<NCEClusterTilingOp>() != nullptr);
        }
        return true;
    });

    target.addLegalOp<NCEClusterTilingOp>();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    func->walk([](mlir::Operation* op) {
        if (op->hasAttr(multiClusterStrategy)) {
            op->removeAttr(multiClusterStrategy);
        }
    });
}

}  // namespace

//
// createWrapVPUOpsInNCEClusterTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createWrapVPUOpsInNCEClusterTilingPass(Logger log) {
    return std::make_unique<WrapVPUOpsInNCEClusterTilingPass>(log);
}
