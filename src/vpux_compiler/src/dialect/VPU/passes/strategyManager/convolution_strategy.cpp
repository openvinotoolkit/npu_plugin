//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

bool ConvolutionStrategy::doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    mlir::ArrayAttr outputAlignmentAttr = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getOutputTensorNumTiles(op, _numClusters, strategy));
    const auto arch = VPU::getArch(origOp.getOperation());
    const auto canUseCMajor =
            VPU::NCEInvariant::isChannelMajorCompatible(arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

    if (!canUseCMajor) {
        const auto activationAlignment = getActivationTensorAlignment(op, strategy);
        if (activationAlignment.hasValue()) {
            activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.getValue());
        }

        const auto outputAlignment = getOutputTensorAlignment(strategy);
        if (outputAlignment.hasValue()) {
            outputAlignmentAttr = getIntArrayAttr(origOp.getContext(), outputAlignment.getValue());
        }
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(origOp.getContext(), weightAlignment.getValue());
    }

    const auto distributedActivationTensorType =
            createDistributedTensorType(origOp, origOp.input(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignmentAttr, strategy);
    const auto distributeddWeightsTensorType =
            createDistributedTensorType(origOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                        weightAlignmentAttr, strategy);
    const auto distributedOutputTensorType = createDistributedTensorType(
            origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles, outputAlignmentAttr, strategy);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributeddWeightsTensorType,
                             distributedOutputTensorType);
}

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool ConvolutionStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));

    if (outputShape[Dims4D::Act::H] < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEConvolutionOp>(op);
    const auto inputShape = getShape(origOp.input());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto KY = filterShape[Dims4D::Filter::KY];

    return isSOHSupportedByDPU(inputShape, KY, _numClusters, false);
}
