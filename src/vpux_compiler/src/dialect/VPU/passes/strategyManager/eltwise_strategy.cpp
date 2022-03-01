//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

bool EltwiseStrategy::doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const {
    auto origOp = mlir::cast<NCEEltwiseOp>(op);
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(op, strategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp, _numClusters, strategy);
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    auto outputTensorNumTiles = getOutputTensorNumTiles(_numClusters, strategy);
    auto distributedInput1TensorType = createDistributedTensorType(
            origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedInput2TensorType = createDistributedTensorType(
            origOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles);
    return true;
    // return origOp.fitIntoCMX(distributedInput1TensorType, distributedInput2TensorType, distributedOutputTensorType);
}
