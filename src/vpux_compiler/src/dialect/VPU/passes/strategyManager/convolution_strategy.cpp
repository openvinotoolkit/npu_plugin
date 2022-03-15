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

bool ConvolutionStrategy::doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    Optional<SmallVector<int64_t>> activationAlignment = None;
    Optional<SmallVector<int64_t>> weightAlignment = None;
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getActivationTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    auto weightsTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getWeightsTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    auto outputTensorNumTiles = getIntArrayAttr(origOp.getContext(),
                                                getOutputTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    activationAlignment = getActivationTensorAlignment(strategy);
    weightAlignment = getWeightsTensorAlignment(strategy);

    if (activationAlignment.hasValue()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.getValue());
        weightAlignmentAttr = getIntArrayAttr(origOp.getContext(), weightAlignment.getValue());
    }

    auto distributedActivationTensorType =
            createDistributedTensorType(origOp, origOp.input(), activationTensorDistributionMode,
                                        activationTensorNumTiles, activationAlignmentAttr, strategy);
    const auto distributeddWeightsTensorType =
            createDistributedTensorType(origOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles,
                                        weightAlignmentAttr, strategy);
    const auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles,
                                        activationAlignmentAttr, strategy);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributeddWeightsTensorType,
                             distributedOutputTensorType);
}
