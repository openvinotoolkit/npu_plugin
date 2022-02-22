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

bool EltwiseStrategy::doesSplitOverHeightLayerFitIntoCMX(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEEltwiseOp>(op);
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, _numClusters, 1}));
    auto weightsTensorDistributionMode = DistributionMode::DUPLICATED;
    auto weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    auto distributedInput1TensorType = createDistributedInputTensorType(
            origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedInput2TensorType = createDistributedInputTensorType(
            origOp, origOp.input2(), weightsTensorDistributionMode, weightTensorNumTiles);
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);

    return origOp.fitIntoCMX(distributedInput1TensorType, distributedInput2TensorType, distributedOutputTensorType);
}
