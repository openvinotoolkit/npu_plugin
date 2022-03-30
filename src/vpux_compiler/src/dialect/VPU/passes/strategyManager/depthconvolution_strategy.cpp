//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

bool DepthConvolutionStrategy::doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, StringRef strategy) const {
    auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEDepthConvolutionOp operation {0}", nceOp->getName());

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    mlir::ArrayAttr outputAlignmentAttr = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getWeightsTensorNumTiles(nceOp, _numClusters, strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getOutputTensorNumTiles(nceOp, _numClusters, strategy));

    const auto activationAlignment = getActivationTensorAlignment(nceOp, strategy);
    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.getValue());
    }

    const auto outputAlignment = getOutputTensorAlignment(strategy);
    if (outputAlignment.hasValue()) {
        outputAlignmentAttr = getIntArrayAttr(origOp.getContext(), outputAlignment.getValue());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(origOp.getContext(), weightAlignment.getValue());
    }

    const auto distributedActivationTensorType =
            createDistributedTensorType(nceOp, origOp.input(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignmentAttr, strategy);
    const auto distributedWeightsTensorType =
            createDistributedTensorType(nceOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                        weightAlignmentAttr, strategy);
    const auto distributedOutputTensorType = createDistributedTensorType(
            nceOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles, outputAlignmentAttr, strategy);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributedWeightsTensorType,
                             distributedOutputTensorType);
}

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool DepthConvolutionStrategy::isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));

    if (outputShape[Dims4D::Act::H] < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEDepthConvolutionOp operation {0}", nceOp->getName());

    const auto inputShape = getShape(origOp.input());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto KY = filterShape[Dims4D::Filter::KY];

    return isSOHSupportedByDPU(inputShape, KY, _numClusters, true);
}
