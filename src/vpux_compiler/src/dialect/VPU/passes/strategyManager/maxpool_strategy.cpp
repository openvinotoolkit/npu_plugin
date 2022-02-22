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
    auto origOp = mlir::cast<NCEMaxPoolOp>(op);
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, _numClusters, 1}));
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedActivationTensorType = createDistributedInputTensorType(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributedOutputTensorType);
}
