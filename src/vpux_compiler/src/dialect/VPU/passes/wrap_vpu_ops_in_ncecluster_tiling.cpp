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

vpux::NDTypeInterface getDistributedOutputTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                     mlir::IntegerAttr numClusters, VPU::MultiClusterStrategy strategy,
                                                     vpux::NDTypeInterface outputTensorType, bool alignForSOH = true) {
    vpux::NDTypeInterface distributedOutputTensorType;
    if (auto sparseOutputType = outputTensorType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseOutputType.getSparsityMap() != nullptr, "Missing sparsity map from sparse type {0}",
                          sparseOutputType);
        VPUX_THROW_UNLESS(sparseOutputType.getStorageElementTable() == nullptr,
                          "Dynamically populated storage element table is not supported");
        auto distributedDataType = getDistributedOutputTypeFromOp(clusteredOp, sparseOutputType.getData(), numClusters);
        auto distributedSMType =
                getDistributedOutputTypeFromOp(clusteredOp, sparseOutputType.getSparsityMap(), numClusters);
        distributedOutputTensorType = VPU::SparseTensorType::get(distributedDataType, distributedSMType);

    } else {
        distributedOutputTensorType = getDistributedOutputTypeFromOp(clusteredOp, outputTensorType, numClusters);
    }

    if (alignForSOH && strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType =
                adjustOutputAlignmentForSOH(clusteredOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.hasValue()) {
            distributedOutputTensorType = newDistributedOutputTensorType.getValue();
        }
    }

    return distributedOutputTensorType;
}

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
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType);

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
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
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);

        if (activationAlignment.hasValue()) {
            activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
        }
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.input(), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
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
                createDistributedCopyIn(clusteredOp, origOp.activationWindow(), activationWindowDistributionMode,
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
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto filterType = origOp.filter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(outputType, numClusters.getInt(), strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.input(), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

    const auto distributedWeightsCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                    weightAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.activationWindow(), activationWindowDistributionMode,
                                    activationWindowNumTiles, nullptr, strategy);

    const auto origOutput = origOp->getResult(0);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    if (origOp.instructionListTable()) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp =
                createDistributedCopyIn(origOp, origOp.instructionListTable(), instructionListTableDistributionMode,
                                        instructionListTableNumTiles, nullptr, strategy);
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
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTableTensorNumTiles(outputType, numClusters.getInt(), strategy));
    const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
    const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.input(), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

    const auto distributedWeightTableCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.weightsTable(), weightsTableTensorDistributionMode,
                                    weightsTableTensorNumTiles, weightAlignmentAttr, strategy);

    const auto distributedActivationWindowCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.activationWindow(), activationWindowDistributionMode,
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
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().getValue();

    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(clusteredOp, origOp.input(), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

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
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.multiClusterStrategy().getValue();
    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.getValue());
    }

    const auto distributedActivationCopyOp1 =
            createDistributedCopyIn(clusteredOp, origOp.input1(), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

    const auto distributedActivationCopyOp2 =
            createDistributedCopyIn(clusteredOp, origOp.input2(), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

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
    NCESWRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::SWOpInterface>(ctx), _numClusters(numClusters), _log(log) {
        setDebugName("NCESWRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SWOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
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
    const auto strategy = clusteredOp.getMultiClusterStrategyAttr().getValue();
    const auto origOutput = swOp->getResult(0);
    auto outputTensorType = origOutput.getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    // Output alignment is possibly needed to keep compatibility and avoid spilling
    // Only support:
    //       NCE_SW  (Clustering/SOK)
    //          |
    //       NCE_DPU (non SOH/SOHOverlapped)
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType, /*alignForSOH=*/false);

    // TODO: extend to support multiple inputs
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    // Input alignment is possibly needed to keep compatibility and avoid spilling
    // Only support:
    //       NCE_DPU (non SOH/SOHOverlapped)
    //          |
    //       NCE_SW  (Clustering/SOK)
    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(swOp.getContext(), activationAlignment.getValue());
    }

    const auto distributedActivationCopyOp =
            createDistributedCopyIn(clusteredOp, swOp->getOperand(0), activationTensorDistributionMode,
                                    activationTensorNumTiles, activationAlignmentAttr, strategy);

    const auto bodyBuilder = [swOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(swOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*swOp, mapper);
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            swOp->getLoc(), distributedOutputTensorType, mlir::ValueRange{distributedActivationCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(swOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEPermuteQuantizeRewriter
//

class NCEPermuteQuantizeRewriter final : public mlir::OpRewritePattern<NCEPermuteQuantizeOp> {
public:
    NCEPermuteQuantizeRewriter(mlir::MLIRContext* ctx, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<NCEPermuteQuantizeOp>(ctx), _numClusters(numClusters), _log(log) {
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
    VPU::DistributedTensorType composeDistributedType(const VPU::DistributedTensorType distType,
                                                      const vpux::NDTypeInterface ndType,
                                                      const mlir::ArrayAttr tileOverDim,
                                                      const mlir::ArrayAttr fusedKernel,
                                                      const mlir::ArrayAttr fusedStrides,
                                                      const PaddingAttr fusedPads) const;
    mlir::Type fusePaddings(const VPU::DistributedTensorType distType, mlir::Operation* nextConv) const;
    VPU::WorkloadCastOp buildCast(NCEClusterTilingOp copyOp, const vpux::NDTypeInterface targetType,
                                  const mlir::ArrayAttr tileOverDim, mlir::PatternRewriter& rewriter) const;
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult NCEPermuteQuantizeRewriter::matchAndRewrite(NCEPermuteQuantizeOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    if (!origOp.multiClusterStrategy().hasValue()) {
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
    const auto workloadStrategy = origOp.multiClusterStrategy().getValue();
    auto outputTensorType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    // outputTensorType has shape 1x32x3x16, therefore width must be considered when setting the number of clusters.
    auto numClusters =
            VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::W], workloadStrategy);

    mlir::ArrayAttr alignmentAttr = nullptr;
    const auto alignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (alignment.hasValue()) {
        alignmentAttr = getIntArrayAttr(ctx, alignment.getValue());
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
    const auto inputDistType = VPU::createDistributedTensorType(clusteredOp, inputNdType, distMode, tileOverHeight,
                                                                numClusters, alignmentAttr);
    const auto fusedDistType = fusePaddings(inputDistType, nextConv);
    const auto inputCopyOp = buildInputCopy(clusteredOp, inReshape->getOperand(0), fusedDistType);
    const auto castInput = buildCast(inputCopyOp, permuteQuantizeInType, tileOverWidth, rewriter);

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
    const auto workloadOutputTensorType =
            composeDistributedType(workloadInputType, origOutputType, tileOverWidth, nullptr, nullptr, nullptr);
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
    const auto castOutput = buildCast(clusterTilingOp, affineReshapeOutType, tileOverHeight, rewriter);
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
    const auto kernelAttr = distTensorType.kernel();
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
    const auto stridesAttr = distTensorType.strides();
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
    const auto padsAttr = distTensorType.pads();
    if (padsAttr != nullptr) {
        return padsAttr;
    }
    return VPU::getPaddingAttr(distTensorType.getContext(), 0, 0, 0, 0);
}

mlir::Operation* NCEPermuteQuantizeRewriter::getNextCompressConv(mlir::Operation* nceOp) const {
    const auto arch = VPU::getArch(nceOp);
    if (!nceOp->hasOneUse()) {
        return nullptr;
    }
    mlir::Operation* nextOp = *nceOp->getUsers().begin();
    while (nextOp != nullptr) {
        if (mlir::isa<VPU::ViewLikeOpInterface>(nextOp) && nextOp->hasOneUse()) {
            nextOp = *nextOp->getUsers().begin();
        } else if (mlir::isa<VPU::NCEConvolutionOp>(nextOp) && VPU::NCEInvariant::isCompressConvolution(arch, nextOp)) {
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

VPU::DistributedTensorType NCEPermuteQuantizeRewriter::composeDistributedType(const VPU::DistributedTensorType distType,
                                                                              const vpux::NDTypeInterface ndType,
                                                                              const mlir::ArrayAttr tileOverDim,
                                                                              const mlir::ArrayAttr fusedKernel,
                                                                              const mlir::ArrayAttr fusedStrides,
                                                                              const PaddingAttr fusedPads) const {
    auto* ctx = distType.getContext();
    // Update distributed activation attribute.
    const auto origDistTensorAttr = distType.getDistribution();
    const auto mode = origDistTensorAttr.mode();
    const auto kernel = getKernel(origDistTensorAttr, fusedKernel);
    const auto pads = getPads(origDistTensorAttr, fusedPads);
    const auto strides = getStrides(origDistTensorAttr, fusedStrides);
    const auto numClusters = origDistTensorAttr.num_clusters();
    const auto alignment = origDistTensorAttr.alignment();
    const auto distTensorAttr =
            DistributedTensorAttr::get(mode, tileOverDim, kernel, pads, strides, numClusters, alignment, ctx);

    // Compose DistributedTensorType.
    const Shape shape = ndType.getShape().toValues();
    const auto elemType = ndType.getElementType();
    const auto order = mlir::AffineMapAttr::get(ndType.getDimsOrder().toAffineMap(ctx));
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(ctx, MemoryKind::CMX_NN));

    return VPU::DistributedTensorType::get(ctx, shape.raw(), elemType, order, memSpace, distTensorAttr);
}

mlir::Type NCEPermuteQuantizeRewriter::fusePaddings(const VPU::DistributedTensorType distType,
                                                    mlir::Operation* nextConv) const {
    if (nextConv == nullptr) {
        return distType;
    }
    auto* ctx = distType.getContext();
    // Get kernel and padding parameters for PermuteQuantize from trailing convolution.
    auto conv = mlir::cast<VPU::NCEConvolutionOp>(nextConv);
    const auto kernel = getIntArrayAttr(ctx, conv.getKernelSize());
    const auto strides = getIntArrayAttr(ctx, conv.getStrides());
    const auto pads = conv.getPad();

    const auto origDistTensorAttr = distType.getDistribution();
    const auto tileOverDim = origDistTensorAttr.num_tiles();
    if (auto sparseInputType = distType.dyn_cast<VPU::SparseTensorType>()) {
        const auto dataNdType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
        auto distributedDataType = composeDistributedType(distType, dataNdType, tileOverDim, kernel, strides, pads);
        const auto sparsityNdType = sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>();
        auto distributedSMType = composeDistributedType(distType, sparsityNdType, tileOverDim, kernel, strides, pads);
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, nullptr,
                                          sparseInputType.getIsWeights(), sparseInputType.getCompressionScheme());
    }
    const auto ndType = distType.cast<vpux::NDTypeInterface>();
    return composeDistributedType(distType, ndType, tileOverDim, kernel, strides, pads);
}

VPU::WorkloadCastOp NCEPermuteQuantizeRewriter::buildCast(NCEClusterTilingOp copyOp,
                                                          const vpux::NDTypeInterface targetType,
                                                          const mlir::ArrayAttr tileOverDim,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto loc = copyOp->getLoc();
    const auto castLoc = appendLoc(loc, "cast number of input tiles");
    const auto copyType = copyOp->getResult(0).getType();

    const auto copyDistTensorType = copyType.cast<VPU::DistributedTensorType>();
    const auto castToDistType =
            composeDistributedType(copyDistTensorType, targetType, tileOverDim, nullptr, nullptr, nullptr);
    auto cast = rewriter.create<VPU::WorkloadCastOp>(castLoc, castToDistType, copyOp->getResult(0));
    return cast;
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
    const auto numNCEClusters = nceOp.count();

    LayerStrategyCheckerFactory::instance().registerNCEOpStrategy(func, _log);
    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<NCEConvolutionRewriter>(&ctx, numNCEClusters, _log);
    patterns.add<NCEDepthConvolutionRewriter>(&ctx, numNCEClusters, _log);
    patterns.add<NCEMaxPoolRewriter>(&ctx, numNCEClusters, _log);
    patterns.add<NCEAveragePoolRewriter>(&ctx, numNCEClusters, _log);
    patterns.add<NCEEltwiseRewriter>(&ctx, numNCEClusters, _log);
    if (auto swOp = IE::getAvailableExecutor(module, ExecutorKind::SHAVE_ACT)) {
        patterns.add<NCESWRewriter>(&ctx, swOp.count(), _log);
    }
    patterns.add<NCEPermuteQuantizeRewriter>(&ctx, numNCEClusters, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op)) {
            auto strategy = clusteredOp.getMultiClusterStrategyAttr();
            if (strategy.hasValue()) {
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

std::unique_ptr<mlir::Pass> VPU::createWrapVPUOpsInNCEClusterTilingPass(Logger log) {
    return std::make_unique<WrapVPUOpsInNCEClusterTilingPass>(log);
}
