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
// GenericNCEtoNCEClusterTiling
//

template <class ConcreteOp>
class GenericNCEtoNCEClusterTiling final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericNCEtoNCEClusterTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("NCEtoNCEClusterTiling::GenericConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericNCEtoNCEClusterTiling<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp);
    auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(origOp);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp);
    auto weightTensorNumTiles = getWeightsTensorNumTiles(origOp);
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(origOp);

    auto distributedActivationCopyOp = createDistributedInputTensor(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedWeightsCopyOp =
            createDistributedInputTensor(origOp, origOp.filter(), weightsTensorDistributionMode, weightTensorNumTiles);

    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, outputTensorDistributionMode, activationTensorNumTiles);

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().template cast<vpux::NDTypeInterface>();
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
            mlir::ValueRange{distributedActivationCopyOp->getResult(0), distributedWeightsCopyOp->getResult(0)},
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

template <>
mlir::LogicalResult GenericNCEtoNCEClusterTiling<NCEMaxPoolOp>::matchAndRewrite(NCEMaxPoolOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp);

    auto distributedActivationCopyOp = createDistributedInputTensor(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);

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
            origOp->getLoc(), distributedOutputTensorType, distributedActivationCopyOp->getResult(0),
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

template <>
mlir::LogicalResult GenericNCEtoNCEClusterTiling<NCEEltwiseOp>::matchAndRewrite(NCEEltwiseOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp);

    auto distributedActivationCopyOp1 = createDistributedInputTensor(
            origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedActivationCopyOp2 = createDistributedInputTensor(
            origOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);

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

    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<GenericNCEtoNCEClusterTiling<NCEConvolutionOp>>(&ctx, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<NCEDepthConvolutionOp>>(&ctx, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<NCEMaxPoolOp>>(&ctx, _log);
    patterns.insert<GenericNCEtoNCEClusterTiling<NCEEltwiseOp>>(&ctx, _log);

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
