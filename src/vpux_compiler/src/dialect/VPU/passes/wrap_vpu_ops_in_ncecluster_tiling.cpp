//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/types.hpp"

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

    auto* ctx = origOp->getContext();
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Operation '{0}' cannot be converted to VPU::NCEOpInterface", origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputTensorType, numClusters);
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType = adjustOutputAlignmentForSOH(nceOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.hasValue()) {
            distributedOutputTensorType = newDistributedOutputTensorType.getValue();
        }
    }

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(outputType, numClusters.getInt(), strategy));

    const auto arch = VPU::getArch(origOp.getOperation());
    const auto canUseCMajor =
            VPU::NCEInvariant::isChannelMajorCompatible(arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

    if (!canUseCMajor) {
        const auto activationAlignment = getActivationTensorAlignment(nceOp, strategy);

        if (activationAlignment.hasValue()) {
            activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
        }
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(nceOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(nceOp, origOp.filter(), weightsTensorDistributionMode,
                                                                  weightsTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(nceOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

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
    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};
    if (canUseCMajor) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        auto distributedActivationWindowCopyOp =
                createDistributedCopyIn(nceOp, origOp.activationWindow(), activationWindowDistributionMode,
                                        activationWindowNumTiles, nullptr, strategy);

        distributedCopyOps.push_back(distributedActivationWindowCopyOp.getResult(0));
    }

    if (origOp.instructionListTable()) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp =
                createDistributedCopyIn(origOp, origOp.instructionListTable(), instructionListTableDistributionMode,
                                        instructionListTableNumTiles, nullptr, strategy);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult(0));
    }

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

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

    auto* ctx = origOp->getContext();
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Operation '{0}' cannot be converted to VPU::NCEOpInterface", origOp);

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputTensorType, numClusters);
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType = adjustOutputAlignmentForSOH(nceOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.hasValue()) {
            distributedOutputTensorType = newDistributedOutputTensorType.getValue();
        }
    }

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(outputType, numClusters.getInt(), strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

    const auto activationAlignment = getActivationTensorAlignment(nceOp, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(nceOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(nceOp, origOp.filter(), weightsTensorDistributionMode,
                                                                  weightsTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(nceOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(nceOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy);

    const auto origOutput = origOp->getResult(0);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    if (origOp.instructionListTable()) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp =
                createDistributedCopyIn(origOp, origOp.instructionListTable(), instructionListTableDistributionMode,
                                        instructionListTableNumTiles, nullptr, strategy);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult(0));
    }
    distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));

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
    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(nceOp, clusterTilingOp);
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

    auto* ctx = origOp->getContext();
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Operation '{0}' cannot be converted to VPU::NCEOpInterface", origOp);

    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputTensorType, numClusters);
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType = adjustOutputAlignmentForSOH(nceOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.hasValue()) {
            distributedOutputTensorType = newDistributedOutputTensorType.getValue();
        }
    }

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(outputType, numClusters.getInt(), strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

    const auto activationAlignment = getActivationTensorAlignment(nceOp, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(nceOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(nceOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(nceOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy);

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

    const auto outputCopyOp = createDistributedCopyOut(nceOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEAveragePoolRewriterRewrite
//

class NCEAveragePoolRewriter final : public mlir::OpRewritePattern<NCEAveragePoolOp> {
public:
    NCEAveragePoolRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<NCEAveragePoolOp>(ctx), _numClusters(numClusters), _log(log) {
        setDebugName("NCEAveragePoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult NCEAveragePoolRewriter::matchAndRewrite(NCEAveragePoolOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Operation '{0}' cannot be converted to VPU::NCEOpInterface", origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputTensorType, numClusters);
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType = adjustOutputAlignmentForSOH(nceOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.hasValue()) {
            distributedOutputTensorType = newDistributedOutputTensorType.getValue();
        }
    }

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(nceOp, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(nceOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
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
            origOp->getLoc(), distributedOutputTensorType, mlir::ValueRange{distributedActivationCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(nceOp, clusterTilingOp);
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

    auto* ctx = origOp->getContext();
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(nceOp != nullptr, "Operation '{0}' cannot be converted to VPU::NCEOpInterface", origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();
    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    const auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputTensorType, numClusters);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(origOp.getOperation(), strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.getValue());
    }

    const auto distributedActivationCopyOp1 =
            createDistributedCopyIn(nceOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles,
                                    activationAlignmentAttr, strategy);

    const auto distributedActivationCopyOp2 =
            createDistributedCopyIn(nceOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles,
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

    const auto outputCopyOp = createDistributedCopyOut(nceOp, clusterTilingOp);
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

    LayerStrategyCheckerFactory::instance().registerNCEOpStrategy(func, _log);
    mlir::RewritePatternSet patterns(&ctx);

    patterns.insert<NCEConvolutionRewriter>(&ctx, numClusters, _log);
    patterns.insert<NCEDepthConvolutionRewriter>(&ctx, numClusters, _log);
    patterns.insert<NCEMaxPoolRewriter>(&ctx, numClusters, _log);
    patterns.insert<NCEAveragePoolRewriter>(&ctx, numClusters, _log);
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
