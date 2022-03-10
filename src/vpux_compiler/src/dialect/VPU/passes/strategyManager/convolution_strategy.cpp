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

// // The efficiency calculation that is being performed here can be described as follows.
// // A ratio of the real output tensor volume to the actual computation that occurs on the
// // hardwarefor each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
// // A hardware efficiency constant is multiplied by the result for channel-major convolutions.
// double ConvolutionStrategy::computeSplitOverHeightEfficiency(mlir::Operation* op) const {
//     auto origOp = mlir::cast<NCEConvolutionOp>(op);
//     const auto outputShape = getShape(origOp.output());
//     const auto OC = outputShape[Dims4D::Act::C];
//     const auto OH = outputShape[Dims4D::Act::H];
//     const auto OW = outputShape[Dims4D::Act::W];
//     const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
//     const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
//     const auto KY = filterShape[Dims4D::Filter::KY];
//     const double outputTensorVolume = OC * OH * OW;
//     const double perClusteroutputTensorVolume = (OH / _numClusters) * OW * OC;

//     if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
//         const auto efficiencyConstant = getChannelMajorEfficiencyConstant(KY, strides[0]);

//         auto efficiency =
//                 std::max(outputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape,
//                                                                       DimsOrder::NCHW, splitOverHeightOverlapped),
//                          outputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape,
//                                                                       DimsOrder::NCHW, splitOverHeightOverlapped));

//         return efficiencyConstant * efficiency;
//     }

//     return std::max(perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape,
//                                                                            DimsOrder::NHWC, splitOverHeight),
//                     perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape,
//                                                                            DimsOrder::NHWC, splitOverHeight));
// }

// double ConvolutionStrategy::computeSplitOverKernelEfficiency(mlir::Operation* op) const {
//     auto origOp = mlir::cast<NCEConvolutionOp>(op);
//     const auto outputShape = getShape(origOp.output());
//     const auto OC = outputShape[Dims4D::Act::C];
//     const auto OH = outputShape[Dims4D::Act::H];
//     const auto OW = outputShape[Dims4D::Act::W];
//     const double perClusteroutputTensorVolume = (OC / _numClusters) * OH * OW;

//     return std::max(perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape,
//                                                                            DimsOrder::NHWC, splitOverKernel),
//                     perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape,
//                                                                            DimsOrder::NHWC, splitOverKernel));
// }
