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

// This depthwise convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> DepthConvolutionStrategy::depthwiseEfficiencyTable() const {
    return {{
            {3, {{1, 0.165}, {2, 0.128}, {4, 0.128}, {6, 0.165}}},
            {5, {{1, 0.483}, {2, 0.241}, {4, 0.132}, {6, 0.483}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}, {6, 0.0395}}},
            {9, {{1, 0.8008}, {2, 0.4687}, {4, 0.2266}, {6, 0.8008}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}, {6, 0.9023}}},
    }};
}

double DepthConvolutionStrategy::getDepthwiseEfficiencyConstant(int64_t kernel, int64_t stride) const {
    if (depthwiseEfficiencyTable().count(kernel)) {
        auto table = depthwiseEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return depthwiseEfficiencyTable()[kernel][stride];
        } else {
            VPUX_THROW("The stride size {0} does not exist in the depthwise efficiency table", stride);
        }
    } else {
        VPUX_THROW("The kernel size {0} does not exist in the depthwise efficiency table", kernel);
    }
}

double DepthConvolutionStrategy::computeSplitOverHeightEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEDepthConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const double outputTensorVolume = OC * OH * OW;
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const double efficiencyConstant = getDepthwiseEfficiencyConstant(KY, strides[0]);

    return efficiencyConstant *
           std::max((outputTensorVolume / _numClusters) /
                            getChannelAlignment((getChannelAlignment(std::ceil(OH / _numClusters), _numClusters) *
                                                 getChannelAlignment(OW, _numClusters) *
                                                 getChannelAlignment(OC, _numChannelAlignment)),
                                                _numDPUs),
                    (outputTensorVolume / _numClusters) /
                            getChannelAlignment(
                                    (getChannelAlignment(std::ceil(OH / _numClusters), _numChannelAlignment) *
                                     getChannelAlignment(OW, 1) * getChannelAlignment(OC, _numChannelAlignment)),
                                    _numDPUs));
}

double DepthConvolutionStrategy::computeSplitOverKernelEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEDepthConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    const auto OC = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const double outputTensorVolume = OC * OH * OW;
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const double efficiencyConstant = getDepthwiseEfficiencyConstant(KY, strides[0]);

    return efficiencyConstant *
           std::max((outputTensorVolume / _numClusters) /
                            getChannelAlignment(
                                    (getChannelAlignment(OH, _numClusters) * getChannelAlignment(OW, _numClusters) *
                                     getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                    _numDPUs),
                    (outputTensorVolume / _numClusters) /
                            getChannelAlignment(
                                    (getChannelAlignment(OH, _numChannelAlignment) * getChannelAlignment(OW, 1) *
                                     getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                    _numDPUs));
}

StringRef DepthConvolutionStrategy::getOptimalLayerStrategy(mlir::Operation* op) const {
    double splitOverHeightEfficiency = 0.0;
    double splitOverKernelEfficiency = 0.0;

    if (isOperationSplitOverHeightCompatible(op) && doesLayerFitIntoCMX(op, splitOverHeight)) {
        splitOverHeightEfficiency = computeSplitOverHeightEfficiency(op);
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        splitOverKernelEfficiency = computeSplitOverKernelEfficiency(op);
    }

    if (splitOverHeightEfficiency >= splitOverKernelEfficiency) {
        return splitOverHeight;
    } else {
        return splitOverKernel;
    }
}
