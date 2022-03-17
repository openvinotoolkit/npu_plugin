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
int64_t vpux::VPU::getNumberOfClustersToAvoidAlignment(int64_t outputChannels, int64_t numClustersToUseForLayer) {
    for (int64_t clusters = numClustersToUseForLayer; clusters >= 1; clusters--) {
        auto alignedOutputChannels = alignVal<int64_t>(divUp(outputChannels, clusters), KMB_DPU_CHANNELS_ALIGNMENT);
        int64_t remainder = outputChannels - (clusters - 1) * alignedOutputChannels;
        if (remainder <= 0) {
            numClustersToUseForLayer = numClustersToUseForLayer - 1;
        } else {
            return numClustersToUseForLayer;
        }
    }
    return numClustersToUseForLayer;
}

SmallVector<int64_t> vpux::VPU::getActivationTensorNumTiles(int64_t numClustersAvailableForCompilation,
                                                            StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == splitOverKernel) {
        return {1, 1, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getActivationTensorAlignment(mlir::Operation* op, StringRef strategy,
                                                                       bool needAlignment, mlir::ArrayAttr numTiles) {
    if (strategy == splitOverKernel) {
        return SmallVector<int64_t>({1, 16, 1, 1});
    } else if (strategy == splitOverHeight) {
        auto kernel = getKernelSize(op);
        auto inputShape = getInputShape(op);
        auto alignment = SmallVector<int64_t>(numTiles.size(), 1);
        const auto numTilesArray = parseIntArrayAttr<int64_t>(numTiles);
        if (numTilesArray[Dims4D::Act::H.ind()] > 1 && kernel && needAlignment) {
            const auto kernelArray = parseIntArrayAttr<int64_t>(kernel);
            const auto KY = kernelArray[0];
            if (KY > 1) {
                alignment[Dims4D::Act::H.ind()] = getSOHPerClusterHeightAlignment(inputShape[Dims4D::Act::W]);
            }
        }
        return alignment;
    } else {
        return SmallVector<int64_t>({1, 1, 1, 1});
    }
}

SmallVector<int64_t> vpux::VPU::getOutputTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                                        StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == splitOverKernel) {
        auto OC = getShape(op->getResult(0))[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = getNumberOfClustersToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {1, numClustersToUseForLayer, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "output tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getWeightsTensorNumTiles(mlir::Operation* op,
                                                         int64_t numClustersAvailableForCompilation,
                                                         StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        auto OC = getShape(op->getResult(0))[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = getNumberOfClustersToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
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

SmallVector<int64_t> vpux::VPU::getWeightsTableTensorNumTiles(mlir::Operation* op,
                                                              int64_t numClustersAvailableForCompilation,
                                                              StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        auto OC = getShape(op->getResult(0))[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = getNumberOfClustersToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getActivationWindowTensorNumTiles(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        return {1, 1, 1, 1};
    } else if (strategy == clustering) {
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
    } else if (strategy == splitOverKernel) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getWeightsTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "weights tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getOutputTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverHeight) {
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
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   " activation window tensor",
                   strategy);
    }
}

NCEClusterTilingOp vpux::VPU::createDistributedCopyOut(mlir::Operation* origOp, NCEClusterTilingOp clusterTilingOp) {
    mlir::OpBuilder builder(origOp);
    auto origOutput = origOp->getResult(0);
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

mlir::ArrayAttr vpux::VPU::getKernelSize(mlir::Operation* origOp) {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<NCEDepthConvolutionOp>(origOp)) {
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShape()));
        return getIntArrayAttr(origOp->getContext(),
                               makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
    } else if (auto convolutionOp = mlir::dyn_cast<NCEConvolutionOp>(origOp)) {
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(convolutionOp.rawFilterShape()));
        return getIntArrayAttr(origOp->getContext(),
                               makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
    } else if (auto maxPoolOp = mlir::dyn_cast<NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.kernel_size();
    } else if (auto eltwiseOp = mlir::dyn_cast<NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get kernel size for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

ShapeRef vpux::VPU::getInputShape(mlir::Operation* origOp) {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<NCEDepthConvolutionOp>(origOp)) {
        const auto inputShape = getShape(depthwiseConvolutionOp.input());
        return inputShape;
    } else if (auto convolutionOp = mlir::dyn_cast<NCEConvolutionOp>(origOp)) {
        const auto inputShape = getShape(convolutionOp.input());
        return inputShape;
    } else if (auto maxPoolOp = mlir::dyn_cast<NCEMaxPoolOp>(origOp)) {
        const auto inputShape = getShape(maxPoolOp.input());
        return inputShape;
    } else if (auto eltwiseOp = mlir::dyn_cast<NCEEltwiseOp>(origOp)) {
        const auto inputShape = getShape(eltwiseOp.input1());
        return inputShape;
    } else {
        VPUX_THROW("Attempting to get kernel size for operation {0}, which is not a NCE Task", origOp->getName());
    }
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
