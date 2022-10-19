//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTensorType> EltwiseStrategy::getDistributedTensorType(
        VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<NCEEltwiseOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEEltwiseOp operation {0}", nceOp->getName());

    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    return SmallVector<VPU::DistributedTensorType>{
            getDistributedActivationTypeFromOp(origOp, origOp.input1().getType(), numClusters, strategy),
            getDistributedActivationTypeFromOp(origOp, origOp.input2().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(origOp, origOp.output().getType(), numClusters, strategy)};
}
