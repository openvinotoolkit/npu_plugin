//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
// compatible it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];

    return OH >= _minimumOutputHeightForSOH;
}

// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
// compitable it must have an output channel of at least the number of clusters x 16
// specified for compilation.
// For example for 4 cluster compilation the output channel must be a
// minimum of 4x16=64.
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _numChannelAlignment * _numClusters;
}

bool BaseLayerStrategy::isOperationMultiClusterCompatible(VPU::NCEOpInterface nceOp) const {
    if (isOperationSplitOverHeightCompatible(nceOp) && doesLayerFitIntoCMX(nceOp, splitOverHeight)) {
        return true;
    }

    if (isOperationSplitOverKernelCompatible(nceOp) && doesLayerFitIntoCMX(nceOp, splitOverKernel)) {
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
                            auto bestStrategy = _convolutionStrategy.getOptimalLayerStrategy(origOp.getOperation());
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

// Based on layer properties like input/output shapes or properties of model or subgraph s
// set layer strategy to predefined value
// Return true if such override has happened
bool StrategyManager::overrideStrategyForLayer(StringRef strategy, VPU::NCEOpInterface nceOp) {
    const auto funcType = _func.getType();

    if (funcType.getInputs().size() == 23 && funcType.getResults().size() == 23) {
        // Temporary solution:
        // For model with 23 inputs and 23 outputs disable multiclustering
        // for layers which were initially assigned Clustering or SOH
        if (strategy == splitOverHeight || strategy == clustering) {
            _log.trace("Override multi-cluster strategy decision to '{0}' for layer '{1}' - '{2}'", "NONE",
                       nceOp->getName(), nceOp->getLoc());
            return true;
        }
    }

    return false;
}

void StrategyManager::setLayerStrategy(StringRef strategy, VPU::NCEOpInterface nceOp) {
    if (overrideStrategyForLayer(strategy, nceOp)) {
        return;
    }

    if (strategy == splitOverHeightOverlapped) {
        nceOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(nceOp->getContext(), "SplitOverHeightOverlapped"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}' - '{2}'", strategy, nceOp->getName(),
                   nceOp->getLoc());
    } else if (strategy == splitOverHeight) {
        nceOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(nceOp->getContext(), "SplitOverHeight"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'- '{2}'", strategy, nceOp->getName(),
                   nceOp->getLoc());
    } else if (strategy == splitOverKernel) {
        nceOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(nceOp->getContext(), "SplitOverKernel"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'- '{2}'", strategy, nceOp->getName(),
                   nceOp->getLoc());
    } else if (strategy == clustering) {
        nceOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(nceOp->getContext(), "Clustering"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'- '{2}'", strategy, nceOp->getName(),
                   nceOp->getLoc());
    } else {
        VPUX_THROW("Attempting to assign an invalid strategy to operation {0}", nceOp->getName());
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

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
double BaseLayerStrategy::computeSplitEfficiency(VPU::NCEOpInterface nceOp, StringRef strategy) const {
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(strategy);
    const auto outputTensorNumTiles =
            getIntArrayAttr(nceOp->getContext(), getOutputTensorNumTiles(nceOp, _numClusters, strategy));
    mlir::ArrayAttr outputAlignmentAttr = nullptr;
    const auto outputAlignment = getOutputTensorAlignment(strategy);
    if (outputAlignment.hasValue()) {
        outputAlignmentAttr = getIntArrayAttr(nceOp->getContext(), outputAlignment.getValue());
    }
    const auto distributedOutputTensorType =
            createDistributedTensorType(nceOp, nceOp->getResult(0), outputTensorDistributionMode, outputTensorNumTiles,
                                        outputAlignmentAttr, strategy);

    const auto perClusterShape = distributedOutputTensorType.getLargestCompactShape();
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    return std::max(static_cast<double>(perClusterOutputTensorVolume) /
                            calculateMPEVolume(VPU::MPEMode::MATRIX, perClusterShape),
                    static_cast<double>(perClusterOutputTensorVolume) /
                            calculateMPEVolume(VPU::MPEMode::VECTOR, perClusterShape));
}

StringRef BaseLayerStrategy::getOptimalLayerStrategy(VPU::NCEOpInterface nceOp) const {
    double splitOverHeightEfficiency = 0.0;
    double splitOverKernelEfficiency = 0.0;
    const auto arch = VPU::getArch(nceOp);
    const auto isChannelMajor = (DimsOrder::fromValue(nceOp->getOperand(0)) == DimsOrder::NCHW) &&
                                VPU::NCEInvariant::isChannelMajorCompatible(
                                        arch, nceOp->getOperand(0).getType().template cast<vpux::NDTypeInterface>());

    if (isOperationSplitOverHeightCompatible(nceOp) &&
        (doesLayerFitIntoCMX(nceOp, splitOverHeightOverlapped) || doesLayerFitIntoCMX(nceOp, splitOverHeight))) {
        splitOverHeightEfficiency = computeSplitEfficiency(nceOp, splitOverHeight);
    }

    if (isOperationSplitOverKernelCompatible(nceOp) && doesLayerFitIntoCMX(nceOp, splitOverKernel)) {
        splitOverKernelEfficiency = computeSplitEfficiency(nceOp, splitOverKernel);
    }

    // Compute ammount of clusters so that SOK is compatible
    const auto module = nceOp->template getParentOfType<mlir::ModuleOp>();
    const auto numClustersAvailableForCompilation = IE::getAvailableExecutor(module, ExecutorKind::NCE).count();
    const auto outputChannels =
            nceOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto sokOptimalClusters =
            getNumberOfClustersForSOKToAvoidAlignment(outputChannels, numClustersAvailableForCompilation);

    const auto optimalHeightTiling = [&](void) {
        return isChannelMajor ? splitOverHeightOverlapped : splitOverHeight;
    };
    if (sokOptimalClusters == numClustersAvailableForCompilation) {
        if (splitOverHeightEfficiency >= splitOverKernelEfficiency) {
            return optimalHeightTiling();
        }
        return splitOverKernel;
    }
    // SOK uses less clusters, but it would be still better than if
    // SOH is incompatible.
    if (splitOverHeightEfficiency > 0) {
        return optimalHeightTiling();
    }
    return splitOverKernel;
}
