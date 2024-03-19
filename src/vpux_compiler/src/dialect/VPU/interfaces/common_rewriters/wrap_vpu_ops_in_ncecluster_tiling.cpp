//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/wrap_vpu_ops_in_ncecluster_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

//
// NCEConvolutionRewriter
//

mlir::LogicalResult VPU::NCEConvolutionRewriter::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto arch = VPU::getArch(origOp.getOperation());
    const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(
            arch, origOp.getInput().getType().cast<vpux::NDTypeInterface>());

    if (!canUseCMajor) {
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);

        if (activationAlignment.has_value()) {
            activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
        }
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
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
    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);

        distributedCopyOps.push_back(distributedActivationWindowCopyOp.getResult(0));
    }

    if (origOp.getInstructionListTable() != nullptr) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                origOp, origOp.getInstructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
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
// NCEDepthConvolutionRewriter
//

mlir::LogicalResult VPU::NCEDepthConvolutionRewriter::matchAndRewrite(VPU::NCEDepthConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto origOutput = origOp->getResult(0);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        const auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    }

    if (origOp.getInstructionListTable() != nullptr) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                origOp, origOp.getInstructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult(0));
    }

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
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

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEMaxPoolRewriter
//

mlir::LogicalResult VPU::NCEMaxPoolRewriter::matchAndRewrite(VPU::NCEMaxPoolOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);
    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0)};

    if (origOp.getWeightsTable() != nullptr) {
        const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTableTensorNumTiles = getIntArrayAttr(
                ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

        const auto distributedWeightTableCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
                weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedWeightTableCopyOp->getResult(0));
    }

    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

        const auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    }
    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEAveragePoolRewriter
//

mlir::LogicalResult VPU::NCEAveragePoolRewriter::matchAndRewrite(VPU::NCEAveragePoolOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
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

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEEltwiseRewriter
//

mlir::LogicalResult VPU::NCEEltwiseRewriter::matchAndRewrite(VPU::NCEEltwiseOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();
    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.value());
    }

    SmallVector<mlir::Value> newEltwiseInputs;
    if (origOp.getInput1() == origOp.getInput2()) {
        const auto distributedActivationCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getInput1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        newEltwiseInputs.push_back(distributedActivationCopyOp->getResult(0));
    } else {
        const auto distributedActivationCopyOp1 = createDistributedCopyIn(
                clusteredOp, origOp.getInput1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

        const auto distributedActivationCopyOp2 = createDistributedCopyIn(
                clusteredOp, origOp.getInput2(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        newEltwiseInputs.push_back(distributedActivationCopyOp1->getResult(0));
        newEltwiseInputs.push_back(distributedActivationCopyOp2->getResult(0));
    }

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     newEltwiseInputs, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCESWRewriter
//

mlir::LogicalResult VPU::NCESWRewriter::matchAndRewrite(VPU::SWOpInterface swOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), swOp->getName(), swOp->getLoc());

    if (swOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, swOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface", swOp);

    auto* ctx = swOp->getContext();

    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(swOp->getResult(0))[Dims4D::Act::C], strategy);

    SmallVector<mlir::Value> distributedCopyOps;
    for (auto operand : swOp->getOperands()) {
        const auto operandType = operand.getType().cast<vpux::NDTypeInterface>();
        const auto activationTensorDistributionMode =
                getSWInputTensorDistributionMode(clusteredOp, strategy, operandType);
        const auto activationTensorNumTiles = getIntArrayAttr(
                ctx, getSWInputTensorNumTiles(clusteredOp, numClusters.getInt(), strategy, operandType));

        // Input alignment is possibly needed to keep compatibility and avoid spilling
        // Only support:
        //       NCE_DPU (non SOH/SOHOverlapped)
        //          |
        //       NCE_SW  (Clustering/SOK)
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy, operandType);
        const auto activationAlignmentAttr = activationAlignment.has_value()
                                                     ? getIntArrayAttr(swOp.getContext(), activationAlignment.value())
                                                     : nullptr;

        const auto distributedCopyOp = createDistributedCopyIn(clusteredOp, operand, activationTensorDistributionMode,
                                                               activationTensorNumTiles, activationAlignmentAttr,
                                                               strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedCopyOp->getResult(0));
    }

    SmallVector<mlir::Type> distributedOutputTypes;
    for (const auto& origOutput : swOp->getResults()) {
        _log.trace("[{0}] Got tag: {1}\n", getDebugName(), origOutput);
        auto outputTensorType = origOutput.getType().cast<vpux::NDTypeInterface>();
        // Output alignment is possibly needed to keep compatibility and avoid spilling
        // Only support:
        //       NCE_SW  (Clustering/SOK)
        //          |
        //       NCE_DPU (non SOH/SOHOverlapped)
        auto distributedOutputTensorType = getDistributedOutputTensorType(
                clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr,
                /*alignForSOH=*/false);
        distributedOutputTypes.push_back(distributedOutputTensorType);
    }

    const auto bodyBuilder = [swOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        auto* newOp = builder.clone(*swOp);
        for (auto operand : swOp->getOperands() | indexed) {
            newOp->setOperand(operand.index(), newOperands[operand.index()]);
        }

        for (const auto& result : swOp->getResults() | indexed) {
            auto newOutput = newOp->getResult(result.index());
            const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
            const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
            newOutput.setType(cmxMemSpace);
        }

        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            swOp->getLoc(), mlir::TypeRange{distributedOutputTypes}, mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    SmallVector<mlir::Value> newOutputs;
    for (const auto& result : swOp->getResults() | indexed) {
        const auto index = result.index();
        const auto origOutput = result.value();
        const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
        const auto origOutMemSpace = origOutType.getMemSpace();

        const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                 mlir::ValueRange newOperands) {
            auto outputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], origOutMemSpace);
            builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
        };

        auto outputCopyOp = rewriter.create<NCEClusterTilingOp>(
                clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(index), outputTensorBodyBuilder);

        origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
        newOutputs.push_back(outputCopyOp->getResult(0));
    }

    rewriter.replaceOp(swOp, newOutputs);
    return mlir::success();
}

//
// NCEPermuteQuantizeRewriter
//

mlir::LogicalResult VPU::NCEPermuteQuantizeRewriter::matchAndRewrite(VPU::NCEPermuteQuantizeOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    if (!origOp.getMultiClusterStrategy().has_value()) {
        return matchFailed(_log, rewriter, origOp, "The operation does not have multi-cluster strategy.");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    // NCE PermuteQuantize is expected to be surrounded by reshapes after ConvertIEToVPUNCEPass.
    // For example:
    // Input (1x3x16x32) -> Reshape (1x32x3x16) -> LayoutCast (NCHW to NHWC) -> PermuteQuantize
    // PermuteQuantize input DMA must tile the original shape over height:
    // 1x3x16x32 with two tiles yields 1x3x8x32 shape per tile (split over height).
    // PermuteQuantize operation will be converted into NCE ELTWISE.
    // Each workload of that NCE ELTWISE must process 1x32x3x8 shape (split over width).
    // 1x32x3x8 permutation to NWCH gives an equivalent of 1x3x8x32 NHWC in each tile.
    // Output DMA must copy 1x3x8x32 NHWC from both tiles (split over height).
    const auto inLayoutCast = origOp->getOperand(0).getDefiningOp<VPU::LayoutCastOp>();
    VPUX_THROW_WHEN(inLayoutCast == nullptr, "VPU.NCE.PermuteQuantize producer must be a VPU.LayoutCast");

    const auto inReshape = inLayoutCast->getOperand(0).getDefiningOp<VPU::ReshapeOp>();
    VPUX_THROW_WHEN(inReshape == nullptr, "VPU.Reshape -> VPU.LayoutCast -> VPU.NCE.PermuteQuantize not found");

    const auto nextConv = getNextCompressConv(origOp);
    const auto strategy = nextConv == nullptr ? VPU::MultiClusterStrategy::SplitOverHeight
                                              : VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
    const auto workloadStrategy = origOp.getMultiClusterStrategy().value();
    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    // outputTensorType has shape 1x32x3x16, therefore width must be considered when setting the number of clusters.
    auto numClusters =
            VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::W], workloadStrategy);

    mlir::ArrayAttr alignmentAttr = nullptr;
    const auto alignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (alignment.has_value()) {
        alignmentAttr = getIntArrayAttr(ctx, alignment.value());
    }

    const auto distMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto tileOverWidth =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), workloadStrategy));
    const auto tileOverHeight =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto permuteQuantizeInType = clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();

    // 1. Tile original input over height via NNDMA.
    // 2. Cast tiling from split-over-height to split-over-width.
    // 3. Add ELTWISE cluster task.
    // 4. Cast ELTWISE output from split-over-width to split-over-height.
    // 5. Copy split-over-height output via NNDMA.
    const auto inputNdType = inReshape->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    // PermuteQuantize uses padding to perform in-place expansion of its input tensor.
    // In OVERLAPPED mode the padding impacts NNDMA.
    // For example 3x224x224 tensor with bottom pad = 13 will be split into 3x119x224 + 3x105x224.
    // The expected split is 3x112x224 + 3x112x224.
    // Set neutral padding to configure input NNDMA properly.
    const auto neutralPads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    const auto inputDistType = VPU::createDistributedTensorType(
            clusteredOp, inputNdType, distMode, tileOverHeight, numClusters, alignmentAttr,
            _enableExplicitDistributedTensorAttr, getIntArrayAttr(ctx, origOp.getKernelSizeVal()), neutralPads,
            getIntArrayAttr(ctx, origOp.getStridesVal()));
    // Reevaluate the input distributed type, based on the overlapped params of the subsequent conv
    const auto fusedDistType = fusePaddings(clusteredOp, inputDistType, nextConv);
    const auto inputCopyOp = buildInputCopy(clusteredOp, inReshape->getOperand(0), fusedDistType);
    const auto castInput = buildCast(clusteredOp, inputCopyOp, permuteQuantizeInType, tileOverWidth, rewriter);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    // Workload input has float16 data type with non-expanded shape and NHWC order.
    // Workload output must have quantized data type with expanded shape and NWCH order.
    // All these parameters (data type, shape and order) can be fetched from the output of the original operation.
    const auto workloadInputType = castInput->getResult(0).getType().cast<VPU::DistributedTensorType>();
    const auto origOutputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto equalComputeAndMemoryView = mlir::UnitAttr::get(ctx);
    const auto workloadOutputTensorType =
            composeDistributedType(clusteredOp, workloadInputType, origOutputType, tileOverWidth, nullptr, nullptr,
                                   nullptr, _enableExplicitDistributedTensorAttr, equalComputeAndMemoryView);
    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), workloadOutputTensorType, mlir::ValueRange{castInput->getResult(0)}, bodyBuilder);

    // Target layout and shape must be fetched from trailing LayoutCastOp and AffineReshapeOp operations.
    // Here distributed output of ELTWISE is casted to 1xCxHxW NHWC output.
    // Trailing reshape and layout cast must be removed, otherwise, graph consistency will be broken.
    const auto outLayoutCast = mlir::dyn_cast<VPU::LayoutCastOp>(*origOp->getUsers().begin());
    VPUX_THROW_WHEN(outLayoutCast == nullptr, "VPU.NCE.PermuteQuantize -> VPU.LayoutCast not found");
    const auto outAffineReshape = mlir::dyn_cast<VPU::AffineReshapeOp>(*outLayoutCast->getUsers().begin());
    // Trivial reshapes (1x16x16x16 to 1x16x16x16) may be eliminated from the graph.
    // In that case, get shape from LayoutCastOp.
    const auto& origOutOp = (outAffineReshape != nullptr) ? outAffineReshape : outLayoutCast;

    const auto affineReshapeOutType = origOutOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto castOutput = buildCast(clusteredOp, clusterTilingOp, affineReshapeOutType, tileOverHeight, rewriter);
    const auto outputCopyOp = buildOutputCopy(origOutOp, castOutput);

    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    origOutOp->getResult(0).replaceAllUsesWith(outputCopyOp->getResult(0));
    if (outAffineReshape != nullptr) {
        rewriter.eraseOp(outAffineReshape);
    }
    rewriter.eraseOp(outLayoutCast);
    rewriter.eraseOp(inReshape);
    rewriter.eraseOp(inLayoutCast);

    return mlir::success();
}

NCEClusterTilingOp VPU::NCEPermuteQuantizeRewriter::buildInputCopy(VPU::ClusteredOpInterface clusteredOp,
                                                                   mlir::Value input, mlir::Type distType) const {
    mlir::OpBuilder builder(clusteredOp);
    builder.setInsertionPoint(clusteredOp);
    const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN));
        auto inputTensorDistributedCopyOp =
                builder.create<VPU::CopyOp>(clusteredOp->getLoc(), newOperands[0], memSpace);
        builder.create<YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
    };

    auto distributedInputCopyOp =
            builder.create<NCEClusterTilingOp>(clusteredOp->getLoc(), distType, input, inputTensorBodyBuilder);

    return distributedInputCopyOp;
}

NCEClusterTilingOp VPU::NCEPermuteQuantizeRewriter::buildOutputCopy(mlir::Operation* nceOp,
                                                                    mlir::Operation* clusterTilingOp) const {
    mlir::OpBuilder builder(nceOp);
    auto origOutput = nceOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    return builder.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0),
                                              outputTensorBodyBuilder);
}

mlir::Type VPU::NCEPermuteQuantizeRewriter::fusePaddings(VPU::ClusteredOpInterface permQuantOp,
                                                         const VPU::DistributedTensorType distType,
                                                         mlir::Operation* nextConv) const {
    if (nextConv == nullptr) {
        return distType;
    }
    auto* ctx = distType.getContext();
    // Get kernel and padding parameters for PermuteQuantize from trailing convolution.
    VPUX_THROW_UNLESS(mlir::isa<VPU::NCEConvolutionOp>(nextConv) || mlir::isa<VPU::NCECompressConvolutionOp>(nextConv),
                      "Next Conv is neither NCEConv nor NCECompressConv");

    auto conv = mlir::cast<VPU::NCEOpInterface>(nextConv);
    const auto kernel = getIntArrayAttr(ctx, conv.getKernelSizeVal());
    const auto strides = getIntArrayAttr(ctx, conv.getStridesVal());
    const auto pads = conv.getPad();

    const auto origDistTensorAttr = distType.getDistribution();
    const auto tileOverDim = origDistTensorAttr.getNumTiles();
    if (auto sparseInputType = distType.dyn_cast<VPU::SparseTensorType>()) {
        const auto dataNdType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
        auto distributedDataType = composeDistributedType(permQuantOp, distType, dataNdType, tileOverDim, kernel,
                                                          strides, pads, _enableExplicitDistributedTensorAttr);
        const auto sparsityNdType = sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>();
        auto distributedSMType = composeDistributedType(permQuantOp, distType, sparsityNdType, tileOverDim, kernel,
                                                        strides, pads, _enableExplicitDistributedTensorAttr);
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, nullptr,
                                          sparseInputType.getIsWeights(), sparseInputType.getCompressionScheme());
    }
    const auto ndType = distType.cast<vpux::NDTypeInterface>();
    return composeDistributedType(permQuantOp, distType, ndType, tileOverDim, kernel, strides, pads,
                                  _enableExplicitDistributedTensorAttr);
}

VPU::WorkloadCastOp VPU::NCEPermuteQuantizeRewriter::buildCast(VPU::ClusteredOpInterface permQuantOp,
                                                               NCEClusterTilingOp copyOp,
                                                               const vpux::NDTypeInterface targetType,
                                                               const mlir::ArrayAttr tileOverDim,
                                                               mlir::PatternRewriter& rewriter) const {
    const auto loc = copyOp->getLoc();
    const auto castLoc = appendLoc(loc, "cast number of input tiles");
    const auto copyType = copyOp->getResult(0).getType();

    const auto copyDistTensorType = copyType.cast<VPU::DistributedTensorType>();
    const auto castToDistType = composeDistributedType(permQuantOp, copyDistTensorType, targetType, tileOverDim,
                                                       nullptr, nullptr, nullptr, _enableExplicitDistributedTensorAttr);
    auto cast = rewriter.create<VPU::WorkloadCastOp>(castLoc, castToDistType, copyOp->getResult(0));
    return cast;
}

//
// NCECompressConvolutionRewriter
//

mlir::LogicalResult VPU::NCECompressConvolutionRewriter::matchAndRewrite(VPU::NCECompressConvolutionOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
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
// NCEInterpolateRewriter
//

mlir::LogicalResult VPU::NCEInterpolateRewriter::matchAndRewrite(VPU::NCEInterpolateOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto weightsType = origOp.getWeights().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, weightsType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto weightAlignment = getWeightsTensorAlignment(strategy);
    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeights(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
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

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}
