//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

bool EltwiseStrategy::doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const {
    auto origOp = mlir::cast<NCEEltwiseOp>(op);
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getActivationTensorNumTiles(_numClusters, strategy));
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(origOp.getContext(), getOutputTensorNumTiles(op, _numClusters, strategy));

    const auto distributedInput1TensorType =
            createDistributedTensorType(origOp, origOp.input1(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignmentAttr, strategy);
    const auto distributedInput2TensorType =
            createDistributedTensorType(origOp, origOp.input2(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignmentAttr, strategy);
    const auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles,
                                        activationAlignmentAttr, strategy);
    return origOp.fitIntoCMX(distributedInput1TensorType, distributedInput2TensorType, distributedOutputTensorType);
}
