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

// This channel major convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> ConvolutionStrategy::channelMajorEfficiencyTable() const {
    static const std::map<int64_t, std::map<int64_t, double>> table = {{
            {3, {{1, 0.253}, {2, 0.183594}, {4, 0.183594}}},
            {5, {{1, 0.535156}, {2, 0.2773}, {4, 0.152344}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}}},
    }};
    return table;
}

double ConvolutionStrategy::getChannelMajorEfficiencyConstant(int64_t kernel, int64_t stride) const {
    if (channelMajorEfficiencyTable().count(kernel)) {
        auto table = channelMajorEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return channelMajorEfficiencyTable()[kernel][stride];
        }
        VPUX_THROW("The stride size {0} does not exist in the channel major efficiency table", stride);
    }
    VPUX_THROW("The kernel size {0} does not exist in the channel major efficiency table", kernel);
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardwarefor each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
// A hardware efficiency constant is multiplied by the result for channel-major convolutions.
double ConvolutionStrategy::computeSplitOverHeightEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const double outputTensorVolume = OC * OH * OW;
    const double perClusteroutputTensorVolume = (OH / _numClusters) * OW * OC;

    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
        const auto efficiencyConstant = getChannelMajorEfficiencyConstant(KY, strides[0]);

        auto efficiency = std::max(
                outputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape, DimsOrder::NCHW),
                outputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape, DimsOrder::NCHW));

        return efficiencyConstant * efficiency;
    }

    return std::max(
            perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape, DimsOrder::NHWC),
            perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape, DimsOrder::NHWC));
}

double ConvolutionStrategy::computeSplitOverKernelEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const double perClusteroutputTensorVolume = (OC / _numClusters) * OH * OW;

    return std::max(
            perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::MATRIX, outputShape, DimsOrder::NHWC),
            perClusteroutputTensorVolume / calculateMPEComputation(VPU::MPEMode::VECTOR, outputShape, DimsOrder::NHWC));
}

StringRef ConvolutionStrategy::getOptimalLayerStrategy(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const bool isChannelMajor = (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW);
    double splitOverHeightEfficiency = 0.0;
    double splitOverKernelEfficiency = 0.0;

    if (isOperationSplitOverHeightCompatible(op) &&
        (doesLayerFitIntoCMX(op, splitOverHeightOverlapped) || doesLayerFitIntoCMX(op, splitOverHeight))) {
        splitOverHeightEfficiency = computeSplitOverHeightEfficiency(op);
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        splitOverKernelEfficiency = computeSplitOverKernelEfficiency(op);
    }

    if (splitOverHeightEfficiency >= splitOverKernelEfficiency) {
        return isChannelMajor ? splitOverHeightOverlapped : splitOverHeight;
    }
    return splitOverKernel;
}
