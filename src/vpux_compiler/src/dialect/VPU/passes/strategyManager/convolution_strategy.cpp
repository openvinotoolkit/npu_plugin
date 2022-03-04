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
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(strategy);
    const auto activationTensorNumTiles = getIntArrayAttr(
            origOp.getContext(), getActivationTensorNumTiles(origOp.getOperation(), _numClusters, strategy));
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
    return {{
            {3, {{1, 0.253}, {2, 0.183594}, {4, 0.183594}}},
            {5, {{1, 0.535156}, {2, 0.2773}, {4, 0.152344}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}}},
    }};
}

double ConvolutionStrategy::getChannelMajorEfficiencyConstant(int64_t kernel, int64_t stride) const {
    if (channelMajorEfficiencyTable().count(kernel)) {
        auto table = channelMajorEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return channelMajorEfficiencyTable()[kernel][stride];
        } else {
            VPUX_THROW("The stride size {0} does not exist in the channel major efficiency table", stride);
        }
    } else {
        VPUX_THROW("The kernel size {0} does not exist in the channel major efficiency table", kernel);
    }
}

double ConvolutionStrategy::computeSplitOverHeightEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const double outputTensorVolume = OC * OH * OW;

    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
        const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
        const auto KY = filterShape[Dims4D::Filter::KY];
        const auto efficiencyConstant = getChannelMajorEfficiencyConstant(KY, strides[0]);
        return efficiencyConstant *
               std::max(outputTensorVolume /
                                (getChannelAlignment(OH, _numChannelAlignment) * getChannelAlignment(OH, _numDPU) *
                                 getChannelAlignment(OC, _numChannelAlignment)),
                        outputTensorVolume / (getChannelAlignment(OH, _numChannelAlignment * _numClusters) *
                                              getChannelAlignment(OW, _numDPUPerCluster) *
                                              getChannelAlignment(OC, _numChannelAlignment)));
    } else {
        return std::max((outputTensorVolume / _numClusters) /
                                getChannelAlignment((getChannelAlignment(std::ceil(OH / _numClusters), _numClusters) *
                                                     getChannelAlignment(OW, _numClusters) *
                                                     getChannelAlignment(OC, _numChannelAlignment)),
                                                    _numDPUPerCluster),
                        (outputTensorVolume / _numClusters) /
                                getChannelAlignment(
                                        (getChannelAlignment(std::ceil(OH / _numClusters), _numChannelAlignment) *
                                         getChannelAlignment(OW, 1) * getChannelAlignment(OC, _numChannelAlignment)),
                                        _numDPUPerCluster));
    }
}

double ConvolutionStrategy::computeSplitOverKernelEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const double outputTensorVolume = OC * OH * OW;

    return std::max(
            (outputTensorVolume / _numClusters) /
                    getChannelAlignment((getChannelAlignment(OH, _numClusters) * getChannelAlignment(OW, _numClusters) *
                                         getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                        _numDPUPerCluster),
            (outputTensorVolume / _numClusters) /
                    getChannelAlignment((getChannelAlignment(OH, _numChannelAlignment) * getChannelAlignment(OW, 1) *
                                         getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                        _numDPUPerCluster));
}

StringRef ConvolutionStrategy::getOptimalLayerStrategy(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const bool isChannelMajor = (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW);
    double splitOverHeightEfficiency = 0.0;
    double splitOverKernelEfficiency = 0.0;

    if (isOperationSplitOverHeightCompatible(op) &&
        ((isChannelMajor && doesLayerFitIntoCMX(op, splitOverHeightOverlapped)) ||
         (!isChannelMajor && doesLayerFitIntoCMX(op, splitOverHeight)))) {
        splitOverHeightEfficiency = computeSplitOverHeightEfficiency(op);
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        splitOverKernelEfficiency = computeSplitOverKernelEfficiency(op);
    }

    if (splitOverHeightEfficiency >= splitOverKernelEfficiency) {
        return isChannelMajor ? splitOverHeightOverlapped : splitOverHeight;
    } else {
        return splitOverKernel;
    }
}
