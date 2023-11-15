//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> MaxPoolStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEMaxPoolOp operation {0}", nceOp->getName());

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    return SmallVector<VPU::DistributedTypeInterface>{
            getDistributedActivationTypeFromOp(origOp, origOp.input().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(clusteredOp, origOp.output().getType(), numClusters, strategy)};
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool MaxPoolStrategy::isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                                           ShapeRef outputShape) const {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(nceOp->getResult(0));
    }
    if (!BaseLayerStrategy::isOperationSplitOverHeightCompatible(nceOp, outputShape)) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEMaxPoolOp operation {0}", nceOp->getName());

    Shape inputShape = getShape(origOp.input()).toValues();
    // If has custom output shape, infer the input shape
    if (outputShape != getShape(nceOp->getResult(0))) {
        vpux::TileInfo outputTile{outputShape};
        auto computerShape = origOp.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles[0].shape;
    }

    auto inputType = origOp.input().getType().cast<NDTypeInterface>();
    return isSOHSupportedByDPU(inputType, inputShape, _numClusters, true, VPU::getArch(nceOp.getOperation()));
}
