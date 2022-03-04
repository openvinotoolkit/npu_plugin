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

// An operation is SOK compitable if it has an output channel of at least 64
// The reason is because the output tensor in each cluster will only have a
// channel of 16 (64/4, assuming 4 cluster compilation).
// There are 5 DPUs in a cluster so each DPU will compute at least one output channel
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _minimumOutputChannelsPerCluster * _numClusters;
}

bool BaseLayerStrategy::isOperationMultiClusterCompatible(mlir::Operation* op) const {
    if (isOperationSplitOverHeightCompatible(op) || isOperationSplitOverKernelCompatible(op)) {
        return true;
    }

    return false;
}

double BaseLayerStrategy::splitOverHeightFormula(double OH, double OW, double OC) const {
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

double BaseLayerStrategy::channelMajorSplitOverHeightFormula(double OH, double OW, double OC) const {
    double outputTensorVolume = OC * OH * OW;
    return std::max(
            outputTensorVolume / (getChannelAlignment(OH, _numChannelAlignment) * getChannelAlignment(OH, _numDPU) *
                                  getChannelAlignment(OC, _numChannelAlignment)),
            outputTensorVolume /
                    (getChannelAlignment(OH, _numChannelAlignment * _numClusters) *
                     getChannelAlignment(OW, _numDPUPerCluster) * getChannelAlignment(OC, _numChannelAlignment)));
}

double BaseLayerStrategy::splitOverKernelFormula(double OH, double OW, double OC) const {
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
                        auto bestStrategy = _convolutionStrategy.getBestLayerStrategy(origOp.getOperation());
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else if (_convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (_depthConvolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy = _depthConvolutionStrategy.getBestLayerStrategy(origOp.getOperation());
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
