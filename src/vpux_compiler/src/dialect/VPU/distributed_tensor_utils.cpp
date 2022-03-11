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

SmallVector<int64_t> vpux::VPU::getActivationTensorNumTiles(mlir::Operation* op, int64_t numClusters,
                                                            StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, numClusters, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, numClusters, 1};
    } else if (strategy == splitOverKernel) {
        if (auto origOp = mlir::dyn_cast<NCEConvolutionOp>(op)) {
            return {1, 1, 1, 1};
        }
        return {1, numClusters, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getOutputTensorNumTiles(int64_t numClusters, StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, numClusters, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, numClusters, 1};
    } else if (strategy == splitOverKernel) {
        return {1, numClusters, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "output tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getWeightsTensorNumTiles(int64_t numClusters, StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        return {numClusters, 1, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getWeightsTableTensorNumTiles(int64_t numClusters, StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        return {numClusters, 1, 1, 1};
    } else if (strategy == clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getActivationWindowTensorNumTiles(int64_t numClusters, StringRef strategy,
                                                                  ArchKind arch) {
    if (strategy == splitOverHeightOverlapped) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverHeight) {
        return {1, 1, 1, 1};
    } else if (strategy == splitOverKernel) {
        if (arch == ArchKind::MTL) {
            return {numClusters, 1, 1, 1};
        } else if (arch == ArchKind::KMB) {
            return {1, 1, 1, 1};
        } else {
            VPUX_THROW("Unsupported arch {0}", arch);
        }
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
        return DistributionMode::SEGMENTED | DistributionMode::DUPLICATED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "output tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getActivationWindowTensorDistributionMode(StringRef strategy, ArchKind arch) {
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverKernel) {
        if (arch == ArchKind::MTL) {
            return DistributionMode::SEGMENTED;
        } else if (arch == ArchKind::KMB) {
            return DistributionMode::DUPLICATED;
        } else {
            VPUX_THROW("Unsupported arch {0}", arch);
        }
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

mlir::ArrayAttr vpux::VPU::getStride(mlir::Operation* origOp) {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<NCEDepthConvolutionOp>(origOp)) {
        return depthwiseConvolutionOp.strides();
    } else if (auto convolutionOp = mlir::dyn_cast<NCEConvolutionOp>(origOp)) {
        return convolutionOp.strides();
    } else if (auto maxPoolOp = mlir::dyn_cast<NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.strides();
    } else if (auto eltwiseOp = mlir::dyn_cast<NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get strides for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

PaddingAttr vpux::VPU::getPad(mlir::Operation* origOp) {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<NCEDepthConvolutionOp>(origOp)) {
        return depthwiseConvolutionOp.padAttr();
    } else if (auto convolutionOp = mlir::dyn_cast<NCEConvolutionOp>(origOp)) {
        return convolutionOp.padAttr();
    } else if (auto maxPoolOp = mlir::dyn_cast<NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.padAttr();
    } else if (auto eltwiseOp = mlir::dyn_cast<NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get pad for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

DistributedTensorType vpux::VPU::createDistributedTensorType(mlir::Operation* origOp, mlir::Value input,
                                                             DistributionMode distributionMode,
                                                             mlir::ArrayAttr numTiles) {
    DistributedTensorAttr distributedActivationTensorAttr;
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClusters = getIntAttr(origOp->getContext(), nceOp.count());
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(origOp->getContext(), distributionMode);

    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto kernel = getKernelSize(origOp);
        auto stride = getStride(origOp);
        auto pad = getPad(origOp);

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           numClusters, nullptr, origOp->getContext());
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           numClusters, nullptr, origOp->getContext());
    } else {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           numClusters, nullptr, origOp->getContext());
    }

    const auto shape = getShape(input);
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp->getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp->getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp->getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedActivationTensorAttr);
}
