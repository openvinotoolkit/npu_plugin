//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> SWInterpolateStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<VPU::InterpolateOp>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::InterpolateOp operation {0}", clusteredOp->getName());

    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);

    SmallVector<VPU::DistributedTypeInterface> distributedTensorTypes;
    for (auto input : origOp->getOperands()) {
        distributedTensorTypes.push_back(
                getDistributedActivationTypeFromOp(clusteredOp, input.getType(), numClusters, strategy));
    }

    distributedTensorTypes.push_back(
            getDistributedOutputTypeFromOp(clusteredOp, origOp->getResult(0).getType(), numClusters, strategy));

    return distributedTensorTypes;
}
