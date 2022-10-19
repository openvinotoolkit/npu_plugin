//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTensorType> MaxPoolStrategy::getDistributedTensorType(
        VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEMaxPoolOp operation {0}", nceOp->getName());

    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    return SmallVector<VPU::DistributedTensorType>{
            getDistributedActivationTypeFromOp(origOp, origOp.input().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(origOp, origOp.output().getType(), numClusters, strategy)};
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool MaxPoolStrategy::isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const {
    if (!BaseLayerStrategy::isOperationSplitOverHeightCompatible(nceOp)) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEMaxPoolOp operation {0}", nceOp->getName());

    const auto outShape = getShape(origOp.output());
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());

    vpux::TileInfo outputTile{outShape};
    auto computerShape = origOp.backInferTileInfo(outputTile, Logger::global());

    return isSOHSupportedByDPU(computerShape.tiles[0].shape, _numClusters, true);
}
