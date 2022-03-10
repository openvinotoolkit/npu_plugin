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

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
// bool ConvolutionStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
//     const auto outputShape = getShape(op->getResult(0));
//     const auto OH = outputShape[Dims4D::Act::H];

//     if (OH < _minimumOutputHeightForSOH) {
//         return false;
//     }

//     auto origOp = mlir::dyn_cast<NCEConvolutionOp>(op);
//     const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
//     const auto KY = filterShape[Dims4D::Filter::KY];
//     if (KY == 1) {
//         return true;
//     }

//     int64_t multOf8 = 1;
//     constexpr int64_t alignment = 8;

//     while (true) {
//         auto x = OH - (_numClusters - 1) * alignment * multOf8;

//         if (x <= 0) {
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

bool ConvolutionStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];

    if (OH < _minimumOutputHeightForSOH) {
        return false;
    }

    auto origOp = mlir::dyn_cast<NCEConvolutionOp>(op);
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    if (KY == 1) {
        return true;
    }

    const auto inputShape = getShape(origOp.input());
    const auto IH = inputShape[Dims4D::Act::H];
    const auto IW = inputShape[Dims4D::Act::W];

    for (int ih = IH / _numClusters; ih < IH; ih++) {
        if (ih * IW % 4 != 0) {
            continue;
        }

        int ihLastCluster = IH - ih * (_numClusters - 1);
        if (ihLastCluster <= 0) {
            return false;
        }

        if (ihLastCluster < ih) {
            break;
            // return true;
        }
    }

    // How to check if result will be possible to divide between clusters?
    for (int oh = OH / _numClusters; oh < OH; oh++) {
        if (oh * OW % 4 != 0) {
            continue;
        }

        int ohLastCluster = OH - oh * (_numClusters - 1);
        if (ohLastCluster <= 0) {
            return false;
        }

        if (ohLastCluster < oh) {
            return true;
        }
    }

    return false;
}