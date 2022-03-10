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
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles = getIntArrayAttr(
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

// double DepthConvolutionStrategy::computeSplitOverHeightEfficiency(mlir::Operation* op) const {
//     auto origOp = mlir::cast<NCEDepthConvolutionOp>(op);
//     const auto outputShape = getShape(origOp.output());
//     const auto OC = outputShape[Dims4D::Act::C];
//     const auto OH = outputShape[Dims4D::Act::H];
//     const auto OW = outputShape[Dims4D::Act::W];
//     const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
//     const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
//     const auto KY = filterShape[Dims4D::Filter::KY];
//     const double efficiencyConstant = getDepthwiseEfficiencyConstant(KY, strides[0]);
//     const double perClusteroutputTensorVolume = (OH / _numClusters) * OW * OC;

//     auto efficiency =
//             std::max(perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape,
//                                                                             DimsOrder::NHWC, splitOverHeight),
//                      perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape,
//                                                                             DimsOrder::NHWC, splitOverHeight));

//     return efficiencyConstant * efficiency;
// }

// double DepthConvolutionStrategy::computeSplitOverKernelEfficiency(mlir::Operation* op) const {
//     auto origOp = mlir::cast<NCEDepthConvolutionOp>(op);
//     const auto outputShape = getShape(origOp.output());
//     const auto OC = outputShape[Dims4D::Act::C];
//     const auto OH = outputShape[Dims4D::Act::H];
//     const auto OW = outputShape[Dims4D::Act::W];
//     const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
//     const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
//     const auto KY = filterShape[Dims4D::Filter::KY];
//     const double efficiencyConstant = getDepthwiseEfficiencyConstant(KY, strides[0]);
//     const double perClusteroutputTensorVolume = (OC / _numClusters) * OW * OH;

//     auto efficiency =
//             std::max(perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape,
//                                                                             DimsOrder::NHWC, splitOverKernel),
//                      perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape,
//                                                                             DimsOrder::NHWC, splitOverKernel));

//     return efficiencyConstant * efficiency;
// }
