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

bool MaxPoolStrategy::doesSplitOverHeightLayerFitIntoCMX(mlir::Operation* op) const {
    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(op);
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(origOp);
    auto activationTensorNumTiles = getActivationTensorNumTiles(origOp);
    auto distributedOutputTensorType = createDistributedTensorType(
            origOp, origOp.output(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedActivationTensorType = createDistributedTensorType(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributedOutputTensorType);
}

double MaxPoolStrategy::computeSplitOverHeightEfficiency(mlir::Operation*) const {
    return 1;
}
