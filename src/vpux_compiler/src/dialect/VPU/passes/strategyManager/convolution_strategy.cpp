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

double ConvolutionStrategy::computeSplitOverHeightEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    auto OC = outputShape[Dims4D::Act::C];
    auto OH = outputShape[Dims4D::Act::H];
    auto OW = outputShape[Dims4D::Act::W];
    double outputTensorVolume = OC * OH * OW;

    return std::max(
            (outputTensorVolume / _numClusters) /
                    getChannelAlignment(
                            (getChannelAlignment(std::ceil(OH / _numClusters), _numClusters) *
                             getChannelAlignment(OW, _numClusters) * getChannelAlignment(OC, _numChannelAlignment)),
                            _numDPUPerCluster),
            (outputTensorVolume / _numClusters) /
                    getChannelAlignment((getChannelAlignment(std::ceil(OH / _numClusters), _numChannelAlignment) *
                                         getChannelAlignment(OW, 1) * getChannelAlignment(OC, _numChannelAlignment)),
                                        _numDPUPerCluster));
}

double ConvolutionStrategy::computeSplitOverKernelEfficiency(mlir::Operation* op) const {
    auto origOp = mlir::cast<NCEConvolutionOp>(op);
    const auto outputShape = getShape(origOp.output());
    auto OC = outputShape[Dims4D::Act::C];
    auto OH = outputShape[Dims4D::Act::H];
    auto OW = outputShape[Dims4D::Act::W];
    double outputTensorVolume = OC * OH * OW;

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