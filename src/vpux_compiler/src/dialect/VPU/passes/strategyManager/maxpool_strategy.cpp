//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

bool MaxPoolStrategy::doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, StringRef strategy) const {
    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEMaxPoolOp operation {0}", nceOp->getName());

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr outputAlignmentAttr = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
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

    const auto distributedActivationTensorType =
            createDistributedTensorType(nceOp, origOp.input(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignmentAttr, strategy);
    const auto distributedOutputTensorType = createDistributedTensorType(
            nceOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles, outputAlignmentAttr, strategy);
    return origOp.fitIntoCMX(distributedActivationTensorType, distributedOutputTensorType);
}

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool MaxPoolStrategy::isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));

    if (outputShape[Dims4D::Act::H] < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(nceOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::NCEMaxPoolOp operation {0}", nceOp->getName());

    const auto inputShape = getShape(origOp.input());
    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto KY = kernelSize[0];

    return isSOHSupportedByDPU(inputShape, KY, _numClusters, true);
}
