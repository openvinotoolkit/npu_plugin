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
// NCEConvolutionToNCEClusterTilingRewrite
//

class NCEConvolutionToNCEClusterTiling final : public mlir::OpRewritePattern<NCEConvolutionOp> {
public:
    NCEConvolutionToNCEClusterTiling(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<NCEConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("NCEConvolutionToNCEClusterTiling");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult NCEConvolutionToNCEClusterTiling::matchAndRewrite(NCEConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    DistributionMode activationWindowDistributionMode = DistributionMode::NONE;
    mlir::ArrayAttr activationWindowNumTiles;
    NCEClusterTilingOp distributedActivationWindowCopyOp = nullptr;
    NCEClusterTilingOp clusterTilingOp = nullptr;

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder == DimsOrder::NCHW) {
        activationWindowDistributionMode = getActivationWindowTensorDistributionMode(origOp, strategy, _arch);
        activationWindowNumTiles = getActivationWindowTensorNumTiles(origOp, strategy, _arch);
        distributedActivationWindowCopyOp =
                createDistributedTensor(origOp, origOp.activationWindow(), strategy, activationWindowDistributionMode,
                                        activationWindowNumTiles);
    }

    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp, strategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp, strategy);
    auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(origOp, strategy);
    auto weightTensorNumTiles = getWeightsTensorNumTiles(origOp, strategy);
    auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(origOp, strategy);
    auto weightsTableTensorNumTiles = getWeightsTensorNumTiles(origOp, strategy);
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(origOp, strategy);
    auto outputTensorNumTiles = getOutputTensorNumTiles(origOp, strategy);

    auto distributedActivationCopyOp = createDistributedTensor(
            origOp, origOp.input(), strategy, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedWeightsCopyOp = createDistributedTensor(origOp, origOp.filter(), strategy,
                                                            weightsTensorDistributionMode, weightTensorNumTiles);

    auto distributedWeightTableCopyOp = createDistributedTensor(
            origOp, origOp.weightsTable(), strategy, weightsTableTensorDistributionMode, weightsTableTensorNumTiles);

    auto distributedOutputTensorType = createDistributedTensorType(origOp, origOp.output(), strategy,
                                                                   outputTensorDistributionMode, outputTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    if (inOrder == DimsOrder::NCHW) {
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

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType,
                                                            clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEDepthConvolutionToNCEClusterTilingRewrite
//

class NCEDepthConvolutionToNCEClusterTiling final : public mlir::OpRewritePattern<NCEDepthConvolutionOp> {
public:
    NCEDepthConvolutionToNCEClusterTiling(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<NCEDepthConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("NCEDepthConvolutionToNCEClusterTiling");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEDepthConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult NCEDepthConvolutionToNCEClusterTiling::matchAndRewrite(NCEDepthConvolutionOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp, strategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp, strategy);
    auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(origOp, strategy);
    auto weightTensorNumTiles = getWeightsTensorNumTiles(origOp, strategy);
    auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(origOp, strategy);
    auto weightsTableTensorNumTiles = getWeightsTensorNumTiles(origOp, strategy);
    auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(origOp, strategy, _arch);
    auto activationWindowNumTiles = getActivationWindowTensorNumTiles(origOp, strategy, _arch);
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(origOp, strategy);
    auto outputTensorNumTiles = getOutputTensorNumTiles(origOp, strategy);

    auto distributedActivationCopyOp = createDistributedTensor(
            origOp, origOp.input(), strategy, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedWeightsCopyOp = createDistributedTensor(origOp, origOp.filter(), strategy,
                                                            weightsTensorDistributionMode, weightTensorNumTiles);

    auto distributedWeightTableCopyOp = createDistributedTensor(
            origOp, origOp.weightsTable(), strategy, weightsTableTensorDistributionMode, weightsTableTensorNumTiles);

    auto distributedActivationWindowCopyOp = createDistributedTensor(
            origOp, origOp.activationWindow(), strategy, activationWindowDistributionMode, activationWindowNumTiles);

    auto distributedOutputTensorType = createDistributedTensorType(origOp, origOp.output(), strategy,
                                                                   outputTensorDistributionMode, outputTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightsCopyOp->getResult(0),
                             distributedWeightTableCopyOp->getResult(0),
                             distributedActivationWindowCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType,
                                                            clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEMaxPoolToNCEClusterTilingRewrite
//

class NCEMaxPoolToNCEClusterTiling final : public mlir::OpRewritePattern<NCEMaxPoolOp> {
public:
    NCEMaxPoolToNCEClusterTiling(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<NCEMaxPoolOp>(ctx), _arch(arch), _log(log) {
        setDebugName("NCEMaxPoolToNCEClusterTiling");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult NCEMaxPoolToNCEClusterTiling::matchAndRewrite(NCEMaxPoolOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp, strategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp, strategy);
    auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(origOp, strategy);
    auto weightsTableTensorNumTiles = getWeightsTensorNumTiles(origOp, strategy);
    auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(origOp, strategy, _arch);
    auto activationWindowNumTiles = getActivationWindowTensorNumTiles(origOp, strategy, _arch);
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(origOp, strategy);
    auto outputTensorNumTiles = getOutputTensorNumTiles(origOp, strategy);

    auto distributedActivationCopyOp = createDistributedTensor(
            origOp, origOp.input(), strategy, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedWeightTableCopyOp = createDistributedTensor(
            origOp, origOp.weightsTable(), strategy, weightsTableTensorDistributionMode, weightsTableTensorNumTiles);

    auto distributedActivationWindowCopyOp = createDistributedTensor(
            origOp, origOp.activationWindow(), strategy, activationWindowDistributionMode, activationWindowNumTiles);

    auto distributedOutputTensorType = createDistributedTensorType(origOp, origOp.output(), strategy,
                                                                   outputTensorDistributionMode, outputTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightTableCopyOp->getResult(0),
                             distributedActivationWindowCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType,
                                                            clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEEltwiseToNCEClusterTilingRewrite
//

class NCEEltwiseToNCEClusterTiling final : public mlir::OpRewritePattern<NCEEltwiseOp> {
public:
    NCEEltwiseToNCEClusterTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<NCEEltwiseOp>(ctx), _log(log) {
        setDebugName("NCEEltwiseToNCEClusterTiling");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NCEEltwiseToNCEClusterTiling::matchAndRewrite(NCEEltwiseOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp, strategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp, strategy);

    auto distributedActivationCopyOp1 = createDistributedTensor(
            origOp, origOp.input1(), strategy, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedActivationCopyOp2 = createDistributedTensor(
            origOp, origOp.input2(), strategy, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedOutputTensorType = createDistributedTensorType(
            origOp, origOp.output(), strategy, activationTensorDistributionMode, activationTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto cmxMemSpace = origOutType.changeMemSpace(MemoryKind::CMX_NN);
    origOutput.setType(cmxMemSpace);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType,
            mlir::ValueRange{distributedActivationCopyOp1->getResult(0), distributedActivationCopyOp2->getResult(0)},
            bodyBuilder);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    auto outputCopyOp = rewriter.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType,
                                                            clusterTilingOp->getResult(0), outputTensorBodyBuilder);

    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// WrapMultiClusterLayersInNCEClusterTilingPass
//

class WrapMultiClusterLayersInNCEClusterTilingPass final :
        public WrapMultiClusterLayersInNCEClusterTilingBase<WrapMultiClusterLayersInNCEClusterTilingPass> {
public:
    explicit WrapMultiClusterLayersInNCEClusterTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void WrapMultiClusterLayersInNCEClusterTilingPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& ctx = getContext();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<NCEConvolutionToNCEClusterTiling>(&ctx, arch, _log);
    patterns.insert<NCEDepthConvolutionToNCEClusterTiling>(&ctx, arch, _log);
    patterns.insert<NCEMaxPoolToNCEClusterTiling>(&ctx, arch, _log);
    patterns.insert<NCEEltwiseToNCEClusterTiling>(&ctx, _log);

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
// createWrapMultiClusterLayersInNCEClusterTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createWrapMultiClusterLayersInNCEClusterTilingPass(Logger log) {
    return std::make_unique<WrapMultiClusterLayersInNCEClusterTilingPass>(log);
}
