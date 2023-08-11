//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> SWMultiplyStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<VPU::MultiplyOp>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::MultiplyOp operation {0}", clusteredOp->getName());

    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp.output().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C], strategy);

    return SmallVector<VPU::DistributedTypeInterface>{
            getDistributedActivationTypeFromOp(clusteredOp, origOp.input1().getType(), numClusters, strategy),
            getDistributedActivationTypeFromOp(clusteredOp, origOp.input2().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(clusteredOp, origOp.output().getType(), numClusters, strategy)};
}
