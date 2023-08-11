//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> AveragePoolStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<NCEAveragePoolOp>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEAveragePoolOp operation {0}", clusteredOp->getName());

    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    return SmallVector<VPU::DistributedTypeInterface>{
            getDistributedActivationTypeFromOp(origOp, origOp.input().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(origOp, origOp.output().getType(), numClusters, strategy)};
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool AveragePoolStrategy::isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                               ShapeRef outputShape) const {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(clusteredOp->getResult(0));
    }
    if (outputShape[Dims4D::Act::H] < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEAveragePoolOp>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEAveragePoolOp operation {0}", clusteredOp->getName());

    const auto inputShape = getShape(origOp.input());
    return isSOHSupportedByDPU(inputShape, _numClusters, true, VPU::getArch(clusteredOp.getOperation()));
}
