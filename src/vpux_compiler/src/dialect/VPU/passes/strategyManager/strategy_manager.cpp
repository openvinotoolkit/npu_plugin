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

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

BaseLayerStrategy::BaseLayerStrategy(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    auto dpuExec = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));
    _numClusters = nceOp.count();
    _numDPUs = dpuExec.count();
    _minimumOutputHeightForSOH = _numDPUs * _numClusters;
}

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumOutputHeightForSOH;
}

// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
// compitable it must have an output channel of at least the number of clusters x 16
// specified for compilation.
// For example for 4 cluster compilation the output channel must be a
// minimum of 4x16=64.
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _numChannelAlignment * _numClusters;
}

// The function computes the actual per cluster output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double BaseLayerStrategy::calculateMPEVolume(mlir::Operation* op, VPU::MPEMode mpeMode, StringRef strategy) const {
    double mpeHeight = 16;
    double mpeWidth = 1;

    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(op->getContext(), getOutputTensorNumTiles(_numClusters, strategy));
    const auto distributedOutputTensorType =
            createDistributedTensorType(op, op->getResult(0), outputTensorDistributionMode, outputTensorNumTiles);

    auto perClusterShape = distributedOutputTensorType.getLargestCompactShape();
    double perClusterOutputWidth = perClusterShape[Dims4D::Act::W];
    double perClusterOutputHeight = perClusterShape[Dims4D::Act::H];
    double perClusterOutputChannels = perClusterShape[Dims4D::Act::C];

    if (mpeMode == VPU::MPEMode::VECTOR) {
        mpeHeight = 16;
        mpeWidth = 1;
    } else if (mpeMode == VPU::MPEMode::MATRIX) {
        mpeHeight = 4;
        mpeWidth = 4;
    } else {
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
    }

    return _numDPUs * std::ceil((mpeHeight * std::ceil(perClusterOutputHeight / mpeHeight) * mpeWidth *
                                 std::ceil(perClusterOutputWidth / mpeWidth) * _numChannelAlignment *
                                 std::ceil(perClusterOutputChannels / _numChannelAlignment)) /
                                _numDPUs);
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
double BaseLayerStrategy::computeSplitEfficiency(mlir::Operation* op, StringRef strategy,
                                                 double efficiencyConstant) const {
    double perClusterOutputTensorVolume = 0;

    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(op->getContext(), getOutputTensorNumTiles(_numClusters, strategy));
    const auto distributedOutputTensorType =
            createDistributedTensorType(op, op->getResult(0), outputTensorDistributionMode, outputTensorNumTiles);

    const auto perClusterShape = distributedOutputTensorType.getLargestCompactShape();
    perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return efficiencyConstant *
           std::max(perClusterOutputTensorVolume / calculateMPEVolume(op, VPU::MPEMode::MATRIX, strategy),
                    perClusterOutputTensorVolume / calculateMPEVolume(op, VPU::MPEMode::VECTOR, strategy));
}

StringRef BaseLayerStrategy::getOptimalLayerStrategy(mlir::Operation* op) const {
    llvm::Optional<double> efficiencyConstant = None;
    double splitOverHeightEfficiency = 0.0;
    double splitOverKernelEfficiency = 0.0;
    bool isChannelMajor = false;
    double constant = 1.0;

    if (auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(op)) {
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
        const auto KY = filterShape[Dims4D::Filter::KY];
        efficiencyConstant = getDepthwiseEfficiencyConstant(KY, kernelStrides[Dims4D::Strides::X]);
    }

    if (auto origOp = mlir::dyn_cast<NCEConvolutionOp>(op)) {
        isChannelMajor = (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW);
        if (isChannelMajor) {
            const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
            const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
            const auto KY = filterShape[Dims4D::Filter::KY];
            efficiencyConstant = getChannelMajorEfficiencyConstant(KY, kernelStrides[Dims4D::Strides::X]);
        }
    }

    if (efficiencyConstant.hasValue()) {
        constant = efficiencyConstant.getValue();
    }

    if (isOperationSplitOverHeightCompatible(op) &&
        (doesLayerFitIntoCMX(op, splitOverHeightOverlapped) || doesLayerFitIntoCMX(op, splitOverHeight))) {
        splitOverHeightEfficiency = computeSplitEfficiency(op, splitOverHeight, constant);
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        splitOverKernelEfficiency = computeSplitEfficiency(op, splitOverKernel, constant);
    }

    if (splitOverHeightEfficiency >= splitOverKernelEfficiency) {
        return isChannelMajor ? splitOverHeightOverlapped : splitOverHeight;
    }
    return splitOverKernel;
}

bool BaseLayerStrategy::isOperationMultiClusterCompatible(mlir::Operation* op) const {
    if (isOperationSplitOverHeightCompatible(op) && doesLayerFitIntoCMX(op, splitOverHeight)) {
        return true;
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        return true;
    }

    return false;
}

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log)
        : _func(func),
          _log(log),
          _convolutionStrategy(func, log),
          _depthConvolutionStrategy(func, log),
          _maxPoolStrategy(func, log),
          _eltwiseStrategy(func, log) {
}

void StrategyManager::assignMultiClusterStrategy() {
    const auto callback = [this](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<NCEMaxPoolOp>([this](NCEMaxPoolOp origOp) {
                    if (_maxPoolStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                        _maxPoolStrategy.doesLayerFitIntoCMX(origOp.getOperation(), splitOverHeight)) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
                    } else if (_maxPoolStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Case<NCEEltwiseOp>([this](NCEEltwiseOp origOp) {
                    if (_eltwiseStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                        _eltwiseStrategy.doesLayerFitIntoCMX(origOp.getOperation(), splitOverHeight)) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
                    } else if (_eltwiseStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Case<NCEConvolutionOp>([this](NCEConvolutionOp origOp) {
                    if (_convolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy = _convolutionStrategy.getOptimalLayerStrategy(origOp.getOperation());
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else if (_convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (_depthConvolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy = _depthConvolutionStrategy.getOptimalLayerStrategy(origOp.getOperation());
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else if (_depthConvolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Default([this](mlir::Operation* unknownOp) -> void {
                    _log.trace("Operation '{0}' at '{1}' is not supported by the NCE therefore it should not have a "
                               "multi-cluster strategy",
                               unknownOp->getName(), unknownOp->getLoc());
                });
    };

    _func.walk(callback);
}

void StrategyManager::setLayerStrategy(StringRef strategy, mlir::Operation* origOp) const {
    if (strategy == splitOverHeightOverlapped) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeightOverlapped"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverHeight) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverKernel) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverKernel"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == clustering) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "Clustering"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else {
        VPUX_THROW("Attempting to assign an invalid strategy to operation {0}", origOp->getName());
    }
}
