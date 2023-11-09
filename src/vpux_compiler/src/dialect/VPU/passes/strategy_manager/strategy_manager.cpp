//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <unordered_map>

using namespace vpux;
using namespace VPU;

StrategyManager::StrategyManager(mlir::func::FuncOp func, Logger log)
        : _func(func), _log(log), _costModel(func, log), _optimizer(func, log) {
}

void StrategyManager::assignMultiClusterStrategy(bool enableMultiClusterForSWLayer) {
    auto setLayerStrategy = [this](VPU::MultiClusterStrategy strategy, mlir::Operation* op) {
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
            strategy == VPU::MultiClusterStrategy::Clustering ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
            strategy == VPU::MultiClusterStrategy::HKSwitch || strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
            llvm::TypeSwitch<mlir::Operation*, void>(op).Case<ClusteredOpInterface>(
                    [strategy](ClusteredOpInterface clusterOp) {
                        clusterOp.setMultiClusterStrategy(strategy);
                    });

            _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}' - '{2}'", strategy, op->getName(),
                       op->getLoc());
        } else {
            VPUX_THROW("Attempting to assign an invalid strategy {0} to operation {1}", strategy, op->getName());
        }
    };

    const auto callback = [&](mlir::Operation* op) {
        _log.trace("Getting strategy for op {0}", op->getName());
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<NCEMaxPoolOp>([&](NCEMaxPoolOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEAveragePoolOp>([&](NCEAveragePoolOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEEltwiseOp>([&](NCEEltwiseOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEConvolutionOp>([&](NCEConvolutionOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    const auto arch = VPU::getArch(origOp.getOperation());
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        auto bestStrategy =
                                _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
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
                .Case<NCECompressConvolutionOp>([&](NCECompressConvolutionOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (layerStrategyChecker->isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                            setLayerStrategy(VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
                                             origOp.getOperation());
                        } else {
                            auto bestStrategy =
                                    _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                            setLayerStrategy(bestStrategy, origOp.getOperation());
                        }
                    } else {
                        VPUX_THROW("Unsupported input layout {0} to CompressConvolution ",
                                   DimsOrder::fromValue(origOp.input()));
                    }
                })
                .Case<NCEDepthConvolutionOp>([&](NCEDepthConvolutionOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEPermuteQuantizeOp>([&](NCEPermuteQuantizeOp origOp) {
                    const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
                    const auto inputShape = inputType.getShape();
                    // Such configurations cannot be tiled properly.
                    if (inputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has input rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), inputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }
                    constexpr int64_t MIN_DIM_SIZE_FOR_TILING = 2;
                    if (inputShape[Dims4D::Act::W] < MIN_DIM_SIZE_FOR_TILING) {
                        _log.trace("Operation '{0}' at '{1}' has size {2} over tiled dimension in input. Expected to "
                                   "have greater than or equal to: {3}.",
                                   origOp->getName(), origOp->getLoc(), inputShape[Dims4D::Act::W],
                                   MIN_DIM_SIZE_FOR_TILING);
                        return;
                    }

                    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
                    const auto outputShape = outputType.getShape();
                    if (outputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), outputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }
                    if (outputShape[Dims4D::Act::W] < MIN_DIM_SIZE_FOR_TILING) {
                        _log.trace("Operation '{0}' at '{1}' has size {2} over tiled dimension in output. Expected to "
                                   "have greater than or equal to: {3}.",
                                   origOp->getName(), origOp->getLoc(), outputShape[Dims4D::Act::W],
                                   MIN_DIM_SIZE_FOR_TILING);
                        return;
                    }

                    const auto bestStrategy = VPU::MultiClusterStrategy::SplitOverWidth;
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<NCEInterpolateOp>([&](NCEInterpolateOp origOp) {
                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    auto bestStrategy = _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Case<SWOpInterface>([&](SWOpInterface origOp) {
                    if (!enableMultiClusterForSWLayer) {
                        return;
                    }

                    auto inputShape =
                            origOp.getOperation()->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
                    auto outputShape =
                            origOp.getOperation()->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();

                    // Non 4D Tensor or Tensor with larger batch size cannot be tiled properly.
                    // [E90039]MC support for Non 4D Tensor.
                    VPU::MultiClusterStrategy bestStrategy;
                    if (inputShape.front() > SINGLE_BATCH || inputShape.size() != RANK_REQUIRED_FOR_TILING ||
                        outputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace("Operation '{0}' at '{1}' has input shape {2} forcing clustering", origOp->getName(),
                                   origOp->getLoc(), inputShape);
                        bestStrategy = VPU::MultiClusterStrategy::Clustering;
                    } else {
                        const auto& layerStrategyChecker =
                                LayerStrategyCheckerFactory::instance().get(origOp->getName());
                        if (origOp.supportCycleCostCalculation()) {
                            bestStrategy =
                                    _costModel.getOptimalLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                        } else {
                            bestStrategy = VPU::getDefaultLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                        }
                    }
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                    _log.info("SW Operation '{0}' {1} set to {2}", origOp->getName(), origOp->getLoc(), bestStrategy);
                })
                .Case<ConcatOp>([&](ConcatOp origOp) {
                    const auto inputType = origOp.inputs().front().getType().cast<vpux::NDTypeInterface>();
                    const auto inputShape = inputType.getShape();
                    // Currently the distributed tensor only supports the tiling scheme with numTile shape=4
                    // TODO: #E81820
                    constexpr size_t RANK_REQUIRED_FOR_TILING = 4;
                    if (inputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has input rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), inputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }

                    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
                    const auto outputShape = outputType.getShape();
                    if (outputShape.size() != RANK_REQUIRED_FOR_TILING) {
                        _log.trace(
                                "Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3}.",
                                origOp->getName(), origOp->getLoc(), outputShape.size(), RANK_REQUIRED_FOR_TILING);
                        return;
                    }

                    const auto& layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(origOp->getName());
                    auto bestStrategy = VPU::getDefaultLayerStrategy(origOp.getOperation(), layerStrategyChecker);
                    setLayerStrategy(bestStrategy, origOp.getOperation());
                })
                .Default([&](mlir::Operation* unknownOp) -> void {
                    _log.trace("Operation '{0}' does not support multi cluster", unknownOp->getName(),
                               unknownOp->getLoc());
                });
    };

    _func.walk(callback);
}

void StrategyManager::optimizeMulticlusterStrategy() {
    _optimizer.optimizeStrategyAvoidSpillingOnModel();
}

size_t getNumNonConstantOperands(mlir::Operation* op) {
    return std::count_if(op->operand_begin(), op->operand_end(), [](mlir::Value operand) {
        return !mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp());
    });
}

bool hasLayerWithMultipleInputs(mlir::Operation* op) {
    return std::any_of(op->user_begin(), op->user_end(), [](mlir::Operation* user) {
        return getNumNonConstantOperands(user) > 1 || hasLayerWithMultipleInputs(user);
    });
}

// Temporary strategy is assigned to Concat to help strategy optimization. We need to remove it after strategy manager
// pass.
void StrategyManager::removeTemporaryMulticlusterStrategy() {
    const auto callbackConcat = [](VPU::ConcatOp concatOp) {
        concatOp.removeMultiClusterStrategyAttr();
    };
    // E#81901 When assigning clustering strategy to ConvertOps for the following pattern(s) breaks the accuracy
    // ConvertOp            ConvertOp
    //     |                    |
    //     IntermediateOp       |
    //           |              |
    //              Layer (Layers with multiple inputs)
    // For such cases and strategies remove the strategy as a temporary measure
    const auto callbackConvert = [&](VPU::ConvertOp convertOp) {
        if (hasLayerWithMultipleInputs(convertOp)) {
            _log.info("SW Operation '{0}' {1} removed strategy {2}", convertOp->getName(), convertOp->getLoc(),
                      convertOp.multiClusterStrategy());
            convertOp.removeMultiClusterStrategyAttr();
        }
    };
    _func.walk(callbackConvert);
    _func.walk(callbackConcat);
}
