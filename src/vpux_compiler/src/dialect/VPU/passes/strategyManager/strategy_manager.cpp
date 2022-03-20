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
#include "vpux/compiler/utils/strings.hpp"

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
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (_convolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                            auto bestStrategy = _convolutionStrategy.getOptimalLayerStrategy(origOp);
                            setLayerStrategy(bestStrategy, origOp.getOperation());
                        } else if (_convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                            setLayerStrategy(clustering, origOp.getOperation());
                        }
                    } else if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
                        const auto arch = VPU::getArch(origOp.getOperation());
                        const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(
                                arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

                        if (canUseCMajor &&
                            _convolutionStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                            _convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(),
                                                                     splitOverHeightOverlapped)) {
                            setLayerStrategy(splitOverHeightOverlapped, origOp.getOperation());
                        } else if (_convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                            setLayerStrategy(clustering, origOp.getOperation());
                        }
                    } else {
                        VPUX_THROW("Unsupported input layout {0} to convolution ",
                                   DimsOrder::fromValue(origOp.input()));
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (_depthConvolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy = _depthConvolutionStrategy.getOptimalLayerStrategy(origOp);
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

// Based on layer properties like input/output shapes set layer strategy to predefined value
// Return true if such override has happened
bool StrategyManager::overrideStrategyForLayer(StringRef strategy, mlir::Operation* origOp) {
    const auto funcType = _func.getType();

    if (funcType.getInputs().size() == 23 && funcType.getResults().size() == 23) {
        // Temporary solution:
        // For model with 23 inputs and 23 outputs disable multiclustering
        // for layers which were initially assigned Clustering of SOH
        if (strategy == splitOverHeight || strategy == clustering) {
            std::cout << "Override multi-cluster strategy decision to NONE for " << stringifyLocation(origOp->getLoc())
                      << "\n";
            // _log.trace("Override multi-cluster strategy decision to '{0}' for layer '{1}' - '{2}'", "NONE",
            //            origOp->getName(), origOp->getLoc());
            return true;
        }
    }

    return false;
}

void StrategyManager::setLayerStrategy(StringRef strategy, mlir::Operation* origOp) {
    if (overrideStrategyForLayer(strategy, origOp)) {
        return;
    }

    if (strategy == splitOverHeightOverlapped) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeightOverlapped"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}' - '{2}'", strategy, origOp->getName(),
                   origOp->getLoc());
    } else if (strategy == splitOverHeight) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'- '{2}'", strategy, origOp->getName(),
                   origOp->getLoc());
    } else if (strategy == splitOverKernel) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverKernel"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'- '{2}'", strategy, origOp->getName(),
                   origOp->getLoc());
    } else if (strategy == clustering) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "Clustering"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'- '{2}'", strategy, origOp->getName(),
                   origOp->getLoc());
    } else {
        VPUX_THROW("Attempting to assign an invalid strategy to operation {0}", origOp->getName());
    }
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double BaseLayerStrategy::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    if (mpeMode == VPU::MPEMode::VECTOR) {
        mpeHeight = 16;
        mpeWidth = 1;
    } else if (mpeMode == VPU::MPEMode::MATRIX) {
        mpeHeight = 4;
        mpeWidth = 4;
    } else {
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
    }

    return static_cast<double>(_numDPUs * divUp((mpeHeight * divUp(shape[Dims4D::Act::H], mpeHeight) * mpeWidth *
                                                 divUp(shape[Dims4D::Act::W], mpeWidth) * _numChannelAlignment *
                                                 divUp(shape[Dims4D::Act::C], _numChannelAlignment)),
                                                _numDPUs));
}
