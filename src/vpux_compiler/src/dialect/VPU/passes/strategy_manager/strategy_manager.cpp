//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <unordered_map>

using namespace vpux;
using namespace VPU;

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log)
        : _func(func), _log(log), _costModel(func, log), _optimizer(func, log) {
}

void StrategyManager::assignMultiClusterStrategy() {
    const auto callback = [this](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<NCEMaxPoolOp>([this](NCEMaxPoolOp origOp) {
                    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    if (layerStrategyChecker->isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverHeight, origOp.getOperation());
                    } else {
                        setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
                    }
                })
                .Case<NCEAveragePoolOp>([this](NCEAveragePoolOp origOp) {
                    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    if (layerStrategyChecker->isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy =
                                _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else {
                        setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
                    }
                })
                .Case<NCEEltwiseOp>([this](NCEEltwiseOp origOp) {
                    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    if (layerStrategyChecker->isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                        setLayerStrategy(VPU::MultiClusterStrategy::SplitOverHeight, origOp.getOperation());
                    } else {
                        setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
                    }
                })
                .Case<NCEConvolutionOp>([this](NCEConvolutionOp origOp) {
                    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (layerStrategyChecker->isOperationMultiClusterCompatible(origOp.getOperation())) {
                            auto bestStrategy =
                                    _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                            setLayerStrategy(bestStrategy, origOp.getOperation());
                        } else {
                            setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
                        }
                    } else if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
                        const auto arch = VPU::getArch(origOp.getOperation());
                        const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(
                                arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

                        if (canUseCMajor &&
                            layerStrategyChecker->isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                            setLayerStrategy(VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
                                             origOp.getOperation());
                        } else {
                            setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
                        }
                    } else {
                        VPUX_THROW("Unsupported input layout {0} to convolution ",
                                   DimsOrder::fromValue(origOp.input()));
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    if (layerStrategyChecker->isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy =
                                _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else {
                        setLayerStrategy(VPU::MultiClusterStrategy::Clustering, origOp.getOperation());
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

void StrategyManager::optimizeMulticlusterStrategy() {
    _optimizer.optimizeStrategyAvoidSpillingOnModel();
}

void StrategyManager::setLayerStrategy(VPU::MultiClusterStrategy strategy, VPU::NCEOpInterface nceOp) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        nceOp->setAttr(multiClusterStrategy, VPU::MultiClusterStrategyAttr::get(nceOp->getContext(), strategy));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}' - '{2}'", strategy, nceOp->getName(),
                   nceOp->getLoc());
    } else {
        VPUX_THROW("Attempting to assign an invalid strategy to operation {0}", nceOp->getName());
    }
}
