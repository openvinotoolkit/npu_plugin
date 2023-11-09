//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
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

vpux::NDTypeInterface getDistributedOutputTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                     mlir::IntegerAttr numClusters, VPU::MultiClusterStrategy strategy,
                                                     vpux::NDTypeInterface outputTensorType,
                                                     const bool hasExplicitDistributedAttr, bool alignForSOH = true) {
    vpux::NDTypeInterface distributedOutputTensorType;
    if (auto sparseOutputType = outputTensorType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseOutputType.getSparsityMap() != nullptr, "Missing sparsity map from sparse type {0}",
                          sparseOutputType);
        VPUX_THROW_UNLESS(sparseOutputType.getStorageElementTable() == nullptr,
                          "Dynamically populated storage element table is not supported");
        auto distributedDataType = getDistributedOutputTypeFromOp(clusteredOp, sparseOutputType.getData(), numClusters,
                                                                  hasExplicitDistributedAttr);
        auto distributedSMType = getDistributedOutputTypeFromOp(clusteredOp, sparseOutputType.getSparsityMap(),
                                                                numClusters, hasExplicitDistributedAttr);
        distributedOutputTensorType = VPU::SparseTensorType::get(distributedDataType, distributedSMType);

    } else {
        distributedOutputTensorType =
                getDistributedOutputTypeFromOp(clusteredOp, outputTensorType, numClusters, hasExplicitDistributedAttr);
    }

    if (alignForSOH && strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType =
                adjustOutputAlignmentForSOH(clusteredOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.has_value()) {
            distributedOutputTensorType = newDistributedOutputTensorType.value();
        }
    }

    return distributedOutputTensorType;
}

//
// NCEConvolutionRewriterRewrite
//

class NCEConvolutionRewriter final : public mlir::OpRewritePattern<NCEConvolutionOp> {
public:
    NCEConvolutionRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEConvolutionRewriter::matchAndRewrite(NCEConvolutionOp origOp,
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
    const auto strategy = origOp.multiClusterStrategy().value();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
    const auto canUseCMajor =
            VPU::NCEInvariant::isChannelMajorCompatible(arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

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
            clusteredOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

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
        auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.activationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);

        distributedCopyOps.push_back(distributedActivationWindowCopyOp.getResult(0));
    }

    if (origOp.instructionListTable()) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                origOp, origOp.instructionListTable(), instructionListTableDistributionMode,
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
// NCEDepthConvolutionRewriterRewrite
//

class NCEDepthConvolutionRewriter final : public mlir::OpRewritePattern<NCEDepthConvolutionOp> {
public:
    NCEDepthConvolutionRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEDepthConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEDepthConvolutionRewriter");
    }

public:
    bool _enableExplicitDistributedTensorAttr = false;
    mlir::LogicalResult matchAndRewrite(NCEDepthConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NCEDepthConvolutionRewriter::matchAndRewrite(NCEDepthConvolutionOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().value();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);

    const auto origOutput = origOp->getResult(0);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    if (origOp.instructionListTable()) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                origOp, origOp.instructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult(0));
    }

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

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEMaxPoolRewriterRewrite
//

class NCEMaxPoolRewriter final : public mlir::OpRewritePattern<NCEMaxPoolOp> {
public:
    NCEMaxPoolRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEMaxPoolOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEMaxPoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEMaxPoolRewriter::matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().value();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);

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

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEAveragePoolRewriterRewrite
//

class NCEAveragePoolRewriter final : public mlir::OpRewritePattern<NCEAveragePoolOp> {
public:
    NCEAveragePoolRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEAveragePoolOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEAveragePoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEAveragePoolRewriter::matchAndRewrite(NCEAveragePoolOp origOp,
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
    const auto strategy = origOp.multiClusterStrategy().value();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
            clusteredOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

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

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEEltwiseRewriterRewrite
//

class NCEEltwiseRewriter final : public mlir::OpRewritePattern<NCEEltwiseOp> {
public:
    NCEEltwiseRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEEltwiseOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEEltwiseRewriter::matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().value();
    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
    if (origOp.input1() == origOp.input2()) {
        const auto distributedActivationCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        newEltwiseInputs.push_back(distributedActivationCopyOp->getResult(0));
    } else {
        const auto distributedActivationCopyOp1 = createDistributedCopyIn(
                clusteredOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

        const auto distributedActivationCopyOp2 = createDistributedCopyIn(
                clusteredOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        newEltwiseInputs.push_back(distributedActivationCopyOp1->getResult(0));
        newEltwiseInputs.push_back(distributedActivationCopyOp2->getResult(0));
    }

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

class NCESWRewriter final : public mlir::OpInterfaceRewritePattern<VPU::SWOpInterface> {
public:
    NCESWRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::SWOpInterface>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCESWRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SWOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCESWRewriter::matchAndRewrite(VPU::SWOpInterface swOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), swOp->getName(), swOp->getLoc());

    if (swOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, swOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface", swOp);

    auto* ctx = swOp->getContext();

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
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
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
        if (activationAlignment.has_value()) {
            activationAlignmentAttr = getIntArrayAttr(swOp.getContext(), activationAlignment.value());
        }

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

class NCEPermuteQuantizeRewriter final : public mlir::OpRewritePattern<NCEPermuteQuantizeOp> {
public:
    NCEPermuteQuantizeRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEPermuteQuantizeOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEPermuteQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEPermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::ArrayAttr getKernel(DistributedTensorAttr distTensorType, const mlir::ArrayAttr fusedKernel) const;
    mlir::ArrayAttr getStrides(DistributedTensorAttr distTensorType, const mlir::ArrayAttr fusedStrides) const;
    PaddingAttr getPads(DistributedTensorAttr distTensorType, const PaddingAttr fusedPads) const;
    mlir::Operation* getNextCompressConv(mlir::Operation* nceOp) const;
    NCEClusterTilingOp buildInputCopy(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                      mlir::Type distType) const;
    NCEClusterTilingOp buildOutputCopy(mlir::Operation* nceOp, mlir::Operation* clusterTilingOp) const;
    VPU::DistributedTensorType composeDistributedType(VPU::ClusteredOpInterface permQuantOp,
                                                      const VPU::DistributedTensorType distType,
                                                      const vpux::NDTypeInterface ndType,
                                                      const mlir::ArrayAttr tileOverDim,
                                                      const mlir::ArrayAttr fusedKernel,
                                                      const mlir::ArrayAttr fusedStrides, const PaddingAttr fusedPads,
                                                      const mlir::UnitAttr equalComputeAndMemoryView = nullptr) const;
    mlir::Type fusePaddings(VPU::ClusteredOpInterface permQuantOp, const VPU::DistributedTensorType distType,
                            mlir::Operation* nextConv) const;
    VPU::WorkloadCastOp buildCast(VPU::ClusteredOpInterface permQuantOp, NCEClusterTilingOp copyOp,
                                  const vpux::NDTypeInterface targetType, const mlir::ArrayAttr tileOverDim,
                                  mlir::PatternRewriter& rewriter) const;
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEPermuteQuantizeRewriter::matchAndRewrite(NCEPermuteQuantizeOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    if (!origOp.multiClusterStrategy().has_value()) {
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
    const auto workloadStrategy = origOp.multiClusterStrategy().value();
    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
        mlir::BlockAndValueMapping mapper;
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
                                   nullptr, equalComputeAndMemoryView);
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

mlir::ArrayAttr NCEPermuteQuantizeRewriter::getKernel(DistributedTensorAttr distTensorType,
                                                      const mlir::ArrayAttr fusedKernel) const {
    if (fusedKernel != nullptr) {
        return fusedKernel;
    }
    const auto kernelAttr = distTensorType.getKernel();
    if (kernelAttr != nullptr) {
        return kernelAttr;
    }
    const auto neutralKernel = SmallVector<int64_t>{1, 1};
    return getIntArrayAttr(distTensorType.getContext(), neutralKernel);
}

mlir::ArrayAttr NCEPermuteQuantizeRewriter::getStrides(DistributedTensorAttr distTensorType,
                                                       const mlir::ArrayAttr fusedStrides) const {
    if (fusedStrides != nullptr) {
        return fusedStrides;
    }
    const auto stridesAttr = distTensorType.getStrides();
    if (stridesAttr != nullptr) {
        return stridesAttr;
    }
    const auto neutralStrides = SmallVector<int64_t>{1, 1};
    return getIntArrayAttr(distTensorType.getContext(), neutralStrides);
}

PaddingAttr NCEPermuteQuantizeRewriter::getPads(DistributedTensorAttr distTensorType,
                                                const PaddingAttr fusedPads) const {
    if (fusedPads != nullptr) {
        return fusedPads;
    }
    if (distTensorType != nullptr && distTensorType.getPads() != nullptr) {
        return distTensorType.getPads();
    }
    return VPU::getPaddingAttr(distTensorType.getContext(), 0, 0, 0, 0);
}

mlir::Operation* NCEPermuteQuantizeRewriter::getNextCompressConv(mlir::Operation* nceOp) const {
    if (!nceOp->hasOneUse()) {
        return nullptr;
    }
    mlir::Operation* nextOp = *nceOp->getUsers().begin();
    while (nextOp != nullptr) {
        if (mlir::isa<VPU::ViewLikeOpInterface>(nextOp) && nextOp->hasOneUse()) {
            nextOp = *nextOp->getUsers().begin();
        } else if (mlir::isa<VPU::NCECompressConvolutionOp>(nextOp)) {
            return nextOp;
        } else {
            return nullptr;
        }
    }

    return nullptr;
}

NCEClusterTilingOp NCEPermuteQuantizeRewriter::buildInputCopy(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                              mlir::Type distType) const {
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

NCEClusterTilingOp NCEPermuteQuantizeRewriter::buildOutputCopy(mlir::Operation* nceOp,
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

VPU::DistributedTensorType NCEPermuteQuantizeRewriter::composeDistributedType(
        VPU::ClusteredOpInterface permQuantOp, const VPU::DistributedTensorType distType,
        const vpux::NDTypeInterface ndType, const mlir::ArrayAttr tileOverDim, const mlir::ArrayAttr fusedKernel,
        const mlir::ArrayAttr fusedStrides, const PaddingAttr fusedPads,
        const mlir::UnitAttr equalComputeAndMemoryView) const {
    // Update distributed activation attribute.
    const auto origDistTensorAttr = distType.getDistribution();
    const auto mode = origDistTensorAttr.getMode().getValue();
    const auto kernel = getKernel(origDistTensorAttr, fusedKernel);
    const auto pads = getPads(origDistTensorAttr, fusedPads);
    const auto strides = getStrides(origDistTensorAttr, fusedStrides);
    const auto numClusters = origDistTensorAttr.getNumClusters();
    const auto alignment = origDistTensorAttr.getAlignment();

    return createDistributedTensorType(permQuantOp, ndType, mode, tileOverDim, numClusters, alignment,
                                       _enableExplicitDistributedTensorAttr, kernel, pads, strides,
                                       equalComputeAndMemoryView);
}

mlir::Type NCEPermuteQuantizeRewriter::fusePaddings(VPU::ClusteredOpInterface permQuantOp,
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
        auto distributedDataType =
                composeDistributedType(permQuantOp, distType, dataNdType, tileOverDim, kernel, strides, pads);
        const auto sparsityNdType = sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>();
        auto distributedSMType =
                composeDistributedType(permQuantOp, distType, sparsityNdType, tileOverDim, kernel, strides, pads);
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, nullptr,
                                          sparseInputType.getIsWeights(), sparseInputType.getCompressionScheme());
    }
    const auto ndType = distType.cast<vpux::NDTypeInterface>();
    return composeDistributedType(permQuantOp, distType, ndType, tileOverDim, kernel, strides, pads);
}

VPU::WorkloadCastOp NCEPermuteQuantizeRewriter::buildCast(VPU::ClusteredOpInterface permQuantOp,
                                                          NCEClusterTilingOp copyOp,
                                                          const vpux::NDTypeInterface targetType,
                                                          const mlir::ArrayAttr tileOverDim,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto loc = copyOp->getLoc();
    const auto castLoc = appendLoc(loc, "cast number of input tiles");
    const auto copyType = copyOp->getResult(0).getType();

    const auto copyDistTensorType = copyType.cast<VPU::DistributedTensorType>();
    const auto castToDistType =
            composeDistributedType(permQuantOp, copyDistTensorType, targetType, tileOverDim, nullptr, nullptr, nullptr);
    auto cast = rewriter.create<VPU::WorkloadCastOp>(castLoc, castToDistType, copyOp->getResult(0));
    return cast;
}

//
// NCECompressConvolutionRewriterRewrite
//

class NCECompressConvolutionRewriter final : public mlir::OpRewritePattern<NCECompressConvolutionOp> {
public:
    NCECompressConvolutionRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCECompressConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCECompressConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCECompressConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCECompressConvolutionRewriter::matchAndRewrite(NCECompressConvolutionOp origOp,
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
    const auto strategy = origOp.multiClusterStrategy().value();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
            clusteredOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

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

class NCEInterpolateRewriter final : public mlir::OpRewritePattern<NCEInterpolateOp> {
public:
    NCEInterpolateRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEInterpolateOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEInterpolateRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEInterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEInterpolateRewriter::matchAndRewrite(NCEInterpolateOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    const auto strategy = origOp.multiClusterStrategy().value();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto weightsType = origOp.weights().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
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
            clusteredOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.weights(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

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
// WrapVPUOpsInNCEClusterTilingPass
//

class WrapVPUOpsInNCEClusterTilingPass final :
        public WrapVPUOpsInNCEClusterTilingBase<WrapVPUOpsInNCEClusterTilingPass> {
public:
    WrapVPUOpsInNCEClusterTilingPass(Logger log): _enableExplicitDistributedTensorAttr(false) {
        Base::initLogger(log, Base::getArgumentName());
    };

    explicit WrapVPUOpsInNCEClusterTilingPass(bool enableExplicitDistributedTensorAttr, Logger log)
            : _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    void safeRunOnFunc() final;
};

mlir::LogicalResult WrapVPUOpsInNCEClusterTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableExplicitDistributedTensorAttr.hasValue()) {
        _enableExplicitDistributedTensorAttr = enableExplicitDistributedTensorAttr.getValue();
        return mlir::success();
    }

    return mlir::success();
}

//
// safeRunOnModule
//

void WrapVPUOpsInNCEClusterTilingPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    LayerStrategyCheckerFactory::instance().registerClusteredOpStrategy(func, _log);
    mlir::RewritePatternSet patterns(&ctx);

    // Both ACT Shaves and DPUs are grouped together in NCE clusters, in a symmetric manner.
    // Each NCE cluster has the same amount of DPUs and ACT shaves.
    // Thus shaves have the availability for distributing across clusters similar to DPUs.
    patterns.add<NCEConvolutionRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCEDepthConvolutionRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCEMaxPoolRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCEAveragePoolRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCEEltwiseRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCESWRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCEPermuteQuantizeRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCECompressConvolutionRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);
    patterns.add<NCEInterpolateRewriter>(&ctx, _enableExplicitDistributedTensorAttr, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op)) {
            auto strategy = clusteredOp.getMultiClusterStrategy();
            if (strategy.has_value()) {
                return (op->getParentOfType<NCEClusterTilingOp>() != nullptr);
            }
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

std::unique_ptr<mlir::Pass> VPU::createWrapVPUOpsInNCEClusterTilingPass(bool enableExplicitDistributedTensorAttr,
                                                                        Logger log) {
    return std::make_unique<WrapVPUOpsInNCEClusterTilingPass>(enableExplicitDistributedTensorAttr, log);
}
