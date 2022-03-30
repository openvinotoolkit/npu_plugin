//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/distributed_tensor_utils.hpp"

using namespace vpux;
using namespace VPU;

//
// Distributed tensor utilities
//

// This method computes the number of clusters to be used for an individual SOK
// layer such that additional alignment of the per cluster output channels is not required.
// Example: For 80 output channel / 4 clusters = [20, 20, 20, 20] output channels per cluster.
// 20 is not aligned to 16. Therefore, the compiler should only execute this layer on 3 clusters.
// This would result in [32, 32, 16] output channels per cluster.
int64_t vpux::VPU::getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersToUseForLayer) {
    for (int64_t clusters = numClustersToUseForLayer; clusters >= 1; clusters--) {
        auto alignedOutputChannels = alignVal<int64_t>(divUp(outputChannels, clusters), KMB_DPU_CHANNELS_ALIGNMENT);
        int64_t remainder = outputChannels - (clusters - 1) * alignedOutputChannels;
        if (remainder > 0) {
            return clusters;
        }
    }
    return 1;
}

SmallVector<int64_t> vpux::VPU::getActivationTensorNumTiles(int64_t numClustersAvailableForCompilation,
                                                            StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == splitOverKernel || strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getActivationTensorAlignment(VPU::NCEOpInterface nceOp, StringRef strategy) {
    if (strategy == splitOverKernel) {
        return SmallVector<int64_t>{1, 16, 1, 1};
    } else if (strategy == splitOverHeight) {
        if (auto origOp = mlir::dyn_cast<NCEConvolutionOp>(nceOp.getOperation())) {
            const auto kernel = nceOp.getKernelSize();
            VPUX_THROW_UNLESS(kernel.size() == 2, "Kernel size of operation '{0}' must be 2, but got '{1}'", origOp,
                              kernel.size());

            const auto KY = kernel[Dims4D::Kernel::Y.ind()];
            if (KY <= 1) {
                return None;
            }
            const auto inputShape = getShape(origOp.input());
            const auto heightAlignment = getSOHPerClusterHeightAlignment(inputShape[Dims4D::Act::W]);
            if (heightAlignment <= 1) {
                return None;
            }
            return SmallVector<int64_t>{1, 1, heightAlignment, 1};
        }
    }

    return None;
}

SmallVector<int64_t> vpux::VPU::getOutputTensorNumTiles(VPU::NCEOpInterface nceOp,
                                                        int64_t numClustersAvailableForCompilation,
                                                        StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == splitOverKernel) {
        auto OC = getShape(nceOp->getResult(0))[Dims4D::Act::C];
        int64_t numClustersToUseForLayer =
                getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {1, numClustersToUseForLayer, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "output tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getOutputTensorAlignment(StringRef strategy) {
    if (strategy == splitOverKernel) {
        return SmallVector<int64_t>{1, 16, 1, 1};
    }

    return None;
}

SmallVector<int64_t> vpux::VPU::getWeightsTensorNumTiles(VPU::NCEOpInterface nceOp,
                                                         int64_t numClustersAvailableForCompilation,
                                                         StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight || strategy == clustering) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        auto OC = getShape(nceOp->getResult(0))[Dims4D::Act::C];
        int64_t numClustersToUseForLayer =
                getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getWeightsTensorAlignment(StringRef strategy) {
    if (strategy == splitOverKernel) {
        return SmallVector<int64_t>{16, 1, 1, 1};
    }
    return None;
}

SmallVector<int64_t> vpux::VPU::getWeightsTableTensorNumTiles(VPU::NCEOpInterface nceOp,
                                                              int64_t numClustersAvailableForCompilation,
                                                              StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight || strategy == clustering) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        auto OC = getShape(nceOp->getResult(0))[Dims4D::Act::C];
        int64_t numClustersToUseForLayer =
                getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getActivationWindowTensorNumTiles(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight || strategy == splitOverKernel ||
        strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation window tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getActivationTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::OVERLAPPED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverKernel || strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getWeightsTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight || strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "weights tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getOutputTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::DUPLICATED | DistributionMode::SEGMENTED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "output tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getActivationWindowTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped || strategy == splitOverHeight || strategy == splitOverKernel ||
        strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   " activation window tensor",
                   strategy);
    }
}

NCEClusterTilingOp vpux::VPU::createDistributedCopyOut(VPU::NCEOpInterface nceOp, NCEClusterTilingOp clusterTilingOp) {
    mlir::OpBuilder builder(nceOp);
    auto origOutput = nceOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    return builder.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0),
                                              outputTensorBodyBuilder);
}

int64_t vpux::VPU::getSOHPerClusterHeightAlignment(int64_t inputWidth) {
    // W * h_per_cluster must be divisible by 4, thus
    // if W % 4 == 0, then h alignment needs to be 1
    // if W % 2 == 0, then h alignment needs to be 2
    // else h alignment needs to be 4
    if (inputWidth % 4 == 0) {
        return 1;
    } else if (inputWidth % 2 == 0) {
        return 2;
    }

    return 4;
}

// When doing SOH not all combinations are supported by HW in terms of how input is segmented
// Following rules need to be satisfied if KY > 1:
// - height of clusters from 0 to N - 1 must be equal
// - height of last cluster (which stores the remainder) must be <= of height of previous clusters
// Additional requirement if operation is not of depth-wise type (it is needed for CONV but not for DWCONV or MAXPOOL)
// - Width * height_per_cluster (for cluster 0 - N-1) must be multiple of 4
bool vpux::VPU::isSOHSupportedByDPU(ShapeRef inputShape, int64_t KY, int64_t numClusters, bool DWTypeOp) {
    if (KY == 1) {
        return true;
    }

    const auto IH = inputShape[Dims4D::Act::H];
    const auto IW = inputShape[Dims4D::Act::W];

    auto hPerCluster = divUp(IH, numClusters);
    auto alignment = (DWTypeOp ? 1 : getSOHPerClusterHeightAlignment(IW));

    hPerCluster = alignVal(hPerCluster, alignment);

    auto hLastCluster = IH - hPerCluster * (numClusters - 1);

    return (hLastCluster > 0);
}

DistributedTensorType vpux::VPU::createDistributedTensorType(VPU::NCEOpInterface nceOp, mlir::Value input,
                                                             DistributionMode distributionMode,
                                                             mlir::ArrayAttr numTiles, mlir::ArrayAttr alignment,
                                                             StringRef strategy) {
    auto* ctx = nceOp->getContext();
    DistributedTensorAttr distributedActivationTensorAttr;
    auto module = nceOp->getParentOfType<mlir::ModuleOp>();
    auto nceResOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClustersAvailableForCompilation = getIntAttr(ctx, nceResOp.count());
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(ctx, distributionMode);
    auto optimalNumberOfClusters = numClustersAvailableForCompilation;

    // Here the number of clusters to be used for an individual SOK layer is determined
    // such that additional alignment of the per cluster output channels is not required.
    // For example 80 output channels, the weights should only be split on 3 clusters [32, 32, 16].
    // Also when creating the copy-in for the activation we need to ensure that the number
    // of clusters that the input is duplicated to is also 3 clusters in this case.
    // Therefore we use the variable optimalNumberOfClusters for both purposes here, to detemine
    // num_tiles and numClusters for the activations and the weights.
    if (strategy == splitOverKernel) {
        int64_t numClustersToUseForLayer = numClustersAvailableForCompilation.getValue().getSExtValue();
        auto OC = getShape(nceOp->getResult(0))[Dims4D::Act::C];
        numClustersToUseForLayer = getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersToUseForLayer);
        optimalNumberOfClusters = mlir::IntegerAttr::get(getInt64Type(ctx), numClustersToUseForLayer);
    }

    auto kernel = getIntArrayAttr(ctx, nceOp.getKernelSize());
    const auto shape = getShape(input);
    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto stride = getIntArrayAttr(ctx, nceOp.getStrides());
        auto pad = nceOp.getPad();
        optimalNumberOfClusters = getIntAttr(ctx, nceResOp.count());

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           optimalNumberOfClusters, alignment, ctx);
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, ctx);
    } else if (VPU ::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, ctx);

    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
    }

    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(ctx, MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(ctx));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(ctx, shape.raw(), elemType, order, memSpace, distributedActivationTensorAttr);
}

NCEClusterTilingOp vpux::VPU::createDistributedCopyIn(VPU::NCEOpInterface nceOp, mlir::Value input,
                                                      DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                      mlir::ArrayAttr alignment, StringRef strategy) {
    auto inputTensorDistributedTensorType =
            createDistributedTensorType(nceOp, input, distributionMode, numTiles, alignment, strategy);

    mlir::OpBuilder builder(nceOp);
    builder.setInsertionPoint(nceOp);
    const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN));
        auto inputTensorDistributedCopyOp = builder.create<IE::CopyOp>(nceOp->getLoc(), newOperands[0], memSpace);
        builder.create<YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
    };

    auto distributedInputCopyOp = builder.create<NCEClusterTilingOp>(nceOp->getLoc(), inputTensorDistributedTensorType,
                                                                     input, inputTensorBodyBuilder);

    return distributedInputCopyOp;
}
