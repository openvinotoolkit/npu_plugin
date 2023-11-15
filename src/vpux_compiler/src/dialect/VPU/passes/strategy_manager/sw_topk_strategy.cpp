//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> SWTopKStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<VPU::TopKOp>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::TopKOp operation {0}", clusteredOp->getName());

    VPUX_THROW_UNLESS(origOp->getOperands().size() == 1 && origOp->getResults().size() == 2,
                      "Only supports SW layers with '1' input and '2' output but got '{0}' and '{1}'",
                      origOp->getOperands().size(), origOp->getResults().size());

    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp.output_values().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            strategy);

    return SmallVector<VPU::DistributedTypeInterface>{
            getDistributedActivationTypeFromOp(clusteredOp, origOp.input().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(clusteredOp, origOp.output_values().getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(clusteredOp, origOp.target_shape().getType(), numClusters, strategy)};
}
