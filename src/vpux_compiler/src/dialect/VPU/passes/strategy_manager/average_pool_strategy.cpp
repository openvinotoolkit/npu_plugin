//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTensorType> AveragePoolStrategy::getDistributedTensorType(
        VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<NCEAveragePoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEAveragePoolOp operation {0}", nceOp->getName());

    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    return SmallVector<VPU::DistributedTensorType>{
            getDistributedActivationTypeFromOp(origOp, origOp.input().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(origOp, origOp.output().getType(), numClusters, strategy)};
}

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool AveragePoolStrategy::isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));

    if (outputShape[Dims4D::Act::H] < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEAveragePoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEAveragePoolOp operation {0}", nceOp->getName());

    const auto inputShape = getShape(origOp.input());
    return isSOHSupportedByDPU(inputShape, _numClusters, true);
}
