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
    _numClusters = nceOp.count();
}

// An operation is SOH compitable if it has an output height of at least 20
// The reason is because the output tensor in each cluster will only have a
// height of 5 (20/4, assuming 4 cluster compilation).
// There are 5 DPUs in a cluster so each DPU will compute at least one output line
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumOutputHeightForSOH;
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
                        _maxPoolStrategy.doesSplitOverHeightLayerFitIntoCMX(origOp.getOperation())) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
                    }
                })
                .Case<NCEEltwiseOp>([this](NCEEltwiseOp origOp) {
                    if (_eltwiseStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                        _eltwiseStrategy.doesSplitOverHeightLayerFitIntoCMX(origOp.getOperation())) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
                    }
                })
                .Case<NCEConvolutionOp>([this](NCEConvolutionOp origOp) {
                    // For WW10 channel major convolution will not be excecuted in multi-cluster mode
                    // Only z-major convolution will be considered for multi-cluster mode
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (_convolutionStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                            _convolutionStrategy.doesSplitOverHeightLayerFitIntoCMX(origOp.getOperation())) {
                            setLayerStrategy(splitOverHeight, origOp.getOperation());
                        }
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (_depthConvolutionStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                        _depthConvolutionStrategy.doesSplitOverHeightLayerFitIntoCMX(origOp.getOperation())) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
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
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverHeight) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverKernel) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverKernel"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == clustering) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "Clustering"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else {
        VPUX_THROW("Attempting to assing an invalid strategy to operation {0}", origOp->getName());
    }
}
