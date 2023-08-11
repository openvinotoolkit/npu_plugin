//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> EltwiseStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<NCEEltwiseOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEEltwiseOp operation {0}", nceOp->getName());

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);
    return SmallVector<VPU::DistributedTypeInterface>{
            getDistributedActivationTypeFromOp(origOp, origOp.input1().getType(), numClusters, strategy),
            getDistributedActivationTypeFromOp(origOp, origOp.input2().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(clusteredOp, origOp.output().getType(), numClusters, strategy)};
}
