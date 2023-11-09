//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

// It is only for the SW with single input and output.
// For some specific SW like:
// - Multiply Op with 2 inputs and 1 output
// - Interpolate Op with 4 inputs and 3 of those are optional
// For those Ops need to create specific getDistributedTensorType method
SmallVector<VPU::DistributedTypeInterface> SWGeneralStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::SWOpInterface operation {0}", clusteredOp->getName());

    VPUX_THROW_UNLESS(origOp->getOperands().size() == 1 && origOp->getResults().size() == 1,
                      "Only supports SW layers with '1' input and '1' output but got '{0}' and '{1}'",
                      origOp->getOperands().size(), origOp->getResults().size());

    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            strategy);

    return SmallVector<VPU::DistributedTypeInterface>{
            getDistributedActivationTypeFromOp(clusteredOp, origOp->getOperand(0).getType(), numClusters, strategy),
            getDistributedOutputTypeFromOp(clusteredOp, origOp->getResult(0).getType(), numClusters, strategy)};
}

bool SWGeneralStrategy::isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                             ShapeRef customOutputShape) const {
    return checkMCRestrictions(clusteredOp) &&
           BaseLayerStrategy::isOperationSplitOverHeightCompatible(clusteredOp, customOutputShape);
}

bool SWGeneralStrategy::isOperationSplitOverWidthCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                            ShapeRef customOutputShape) const {
    return checkMCRestrictions(clusteredOp) &&
           BaseLayerStrategy::isOperationSplitOverWidthCompatible(clusteredOp, customOutputShape);
}

bool SWGeneralStrategy::isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                             ShapeRef customOutputShape) const {
    return checkMCRestrictions(clusteredOp) &&
           BaseLayerStrategy::isOperationSplitOverKernelCompatible(clusteredOp, customOutputShape);
}

bool SWGeneralStrategy::checkMCRestrictions(VPU::ClusteredOpInterface clusteredOp) const {
    auto module = clusteredOp->getParentOfType<mlir::ModuleOp>();
    if (IE::getAvailableExecutor(module, VPU::ExecutorKind::SHAVE_ACT) == nullptr) {
        return false;
    }

    auto inputShape = getShape(clusteredOp.getOperation()->getOperand(0));
    auto outputShape = getShape(clusteredOp.getOperation()->getResult(0));
    return !(inputShape.front() > SINGLE_BATCH || inputShape.size() != RANK_REQUIRED_FOR_TILING ||
             outputShape.size() != RANK_REQUIRED_FOR_TILING);
}
