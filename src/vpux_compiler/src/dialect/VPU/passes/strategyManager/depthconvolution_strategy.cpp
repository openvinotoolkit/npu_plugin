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

bool DepthConvolutionStrategy::doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const {
    auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(op);
    mlir::ArrayAttr activationAlignment = nullptr;
    mlir::ArrayAttr weightAlignment = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getActivationTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    auto weightsTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    auto outputTensorNumTiles = getIntArrayAttr(origOp.getContext(),
                                                getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    if (strategy == splitOverKernel) {
        activationAlignment = getIntArrayAttr(origOp.getContext(), getActivationTensorAlignment(op, strategy));
        weightAlignment = getIntArrayAttr(origOp.getContext(), getWeightsTensorAlignment(op, strategy));
    }
    auto distributedActivationTensorType =
            createDistributedTensorType(origOp, origOp.input(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignment, strategy);
    const auto distributeddWeightsTensorType = createDistributedTensorType(
            origOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignment, strategy);
    const auto distributedOutputTensorType = createDistributedTensorType(
            origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles, activationAlignment, strategy);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributeddWeightsTensorType,
                             distributedOutputTensorType);
}
