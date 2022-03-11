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
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    auto activationTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getActivationTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
    auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    auto weightsTensorNumTiles = getIntArrayAttr(origOp.getContext(), getWeightsTensorNumTiles(_numClusters, strategy));
    auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    auto outputTensorNumTiles = getIntArrayAttr(origOp.getContext(), getOutputTensorNumTiles(_numClusters, strategy));
    auto distributedActivationTensorType = createDistributedTensorType(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributeddWeightsTensorType =
            createDistributedTensorType(origOp, origOp.filter(), weightsTensorDistributionMode, weightsTensorNumTiles);
    auto distributedOutputTensorType =
            createDistributedTensorType(origOp, origOp.output(), outputTensorDistributionMode, outputTensorNumTiles);
    return origOp.fitIntoCMX(distributedActivationTensorType, distributeddWeightsTensorType,
                             distributedOutputTensorType);
}

// bool DepthConvolutionStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
//     const auto outputShape = getShape(op->getResult(0));
//     const auto OH = outputShape[Dims4D::Act::H];
//     auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(op);
//     const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
//     const auto KY = filterShape[Dims4D::Filter::KY];

//     if (OH < _minimumOutputHeightForSOH) {
//         return false;
//     }

//     int64_t multOf8 = 1;
//     constexpr int64_t alignment = 8;

//     while (true) {
//         auto x = OH - (_numClusters - 1) * alignment * multOf8;

//         if (x < KY) {
//             return false;
//         }

//         if (alignment * multOf8 > x) {
//             return true;
//         }

//         multOf8++;
//     }

//     return false;
//     // return OH >= _minimumOutputHeightForSOH;
// }

bool DepthConvolutionStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];
    if (OH < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(op);
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    if (KY == 1) {
        return true;
    }

    const auto inputShape = getShape(origOp.input());
    const auto IH = inputShape[Dims4D::Act::H];

    for (int ih = IH / _numClusters; ih < IH; ih++) {
        int ihLastCluster = IH - ih * (_numClusters - 1);
        if (ihLastCluster <= KY) {
            return false;
        }

        if (ihLastCluster <= ih) {
            break;
            // return true;
        }
    }

    // How to check if result will be possible to divide between clusters?
    for (int oh = OH / _numClusters; oh < OH; oh++) {
        int ohLastCluster = OH - oh * (_numClusters - 1);
        if (ohLastCluster <= 0) {
            return false;
        }

        if (ohLastCluster <= oh) {
            return true;
        }
    }

    return false;
}
