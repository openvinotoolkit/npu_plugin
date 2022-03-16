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
    mlir::ArrayAttr activationAlignment = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getActivationTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    auto outputTensorNumTiles = getIntArrayAttr(origOp.getContext(),
                                                getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    if (strategy == splitOverKernel) {
        activationAlignment = getIntArrayAttr(
                origOp.getContext(), getActivationTensorAlignment(op, strategy, false, activationTensorNumTiles));
    }
    auto distributedInput1TensorType = createDistributedTensorType(
            origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles, activationAlignment);
    const auto distributedInput2TensorType = createDistributedTensorType(
            origOp, origOp.input2(), activationTensorDistributionMode, activationTensorNumTiles, activationAlignment);
    const auto distributedOutputTensorType = createDistributedTensorType(
            origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles, activationAlignment);
    return origOp.fitIntoCMX(distributedInput1TensorType, distributedInput2TensorType, distributedOutputTensorType);
}
