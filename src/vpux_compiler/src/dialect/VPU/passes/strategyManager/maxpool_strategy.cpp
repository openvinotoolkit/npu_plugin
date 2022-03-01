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

bool MaxPoolStrategy::doesLayerFitIntoCMX(mlir::Operation* op, StringRef strategy) const {
    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(op);
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp, strategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp, strategy);
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(origOp, strategy);
    auto outputTensorNumTiles = getOutputTensorNumTiles(origOp, strategy);
    auto distributedActivationTensorType = createDistributedTensorType(
            origOp, origOp.input(), strategy, activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedOutputTensorType = createDistributedTensorType(origOp, origOp.output(), strategy,
                                                                   outputTensorDistributionMode, outputTensorNumTiles);
    return true;
    // return origOp.fitIntoCMX(distributedActivationTensorType, distributedOutputTensorType);
}
