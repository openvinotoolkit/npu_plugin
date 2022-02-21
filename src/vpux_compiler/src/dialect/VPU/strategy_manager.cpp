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

// This pass assigns a layer to be SOH if the operation is SOH compitable
// i.e the output height is > 20 and the layer fits in CMX when multi-clustered

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log)
        : _func(func),
          _log(log),
          _convolutionStrategy(func, log),
          _depthConvolutionStrategy(func, log),
          _maxPoolStrategy(func, log),
          _eltwiseStrategy(func, log) {
}

BaseLayerStrategy::BaseLayerStrategy(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    _numClusters = nceOp.count();
}

// An operation is SOH compitable if it has an output height of at least 20
// The reason is because the output tensor in each cluster will only have a
// height of 5 (20/4, assuming 4 cluster compilation).
// There are 5 DPUs in a cluster so each DPU will compute at least one output line
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) {
    const auto outputShape = getShape(op->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumOutputHeightForSOH;
}

bool ConvolutionStrategy::doesSplitOverHeightLayerFitIntoCMX(mlir::Operation* op) {
    auto origOp = mlir::dyn_cast<NCEConvolutionOp>(op);
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, _numClusters, 1}));
    auto weightsTensorDistributionMode = DistributionMode::MULTICASTED;
    auto weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedActivationTensorType = createDistributedInputTensorType(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributeddWeightsTensorType = createDistributedInputTensorType(
            origOp, origOp.filter(), weightsTensorDistributionMode, weightTensorNumTiles);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributeddWeightsTensorType,
                             distributedOutputTensorType);
}

bool MaxPoolStrategy::doesSplitOverHeightLayerFitIntoCMX(mlir::Operation* op) {
    auto origOp = mlir::dyn_cast<NCEMaxPoolOp>(op);
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, _numClusters, 1}));
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);

    auto distributedActivationTensorType = createDistributedInputTensorType(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributedOutputTensorType);
}

bool DepthConvolutionStrategy::doesSplitOverHeightLayerFitIntoCMX(mlir::Operation* op) {
    auto origOp = mlir::dyn_cast<NCEDepthConvolutionOp>(op);
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, _numClusters, 1}));
    auto weightsTensorDistributionMode = DistributionMode::MULTICASTED;
    auto weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedActivationTensorType = createDistributedInputTensorType(
            origOp, origOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributeddWeightsTensorType = createDistributedInputTensorType(
            origOp, origOp.filter(), weightsTensorDistributionMode, weightTensorNumTiles);

    return origOp.fitIntoCMX(distributedActivationTensorType, distributeddWeightsTensorType,
                             distributedOutputTensorType);
}

bool EltwiseStrategy::doesSplitOverHeightLayerFitIntoCMX(mlir::Operation* op) {
    auto origOp = mlir::dyn_cast<NCEEltwiseOp>(op);
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, _numClusters, 1}));
    auto weightsTensorDistributionMode = DistributionMode::MULTICASTED;
    auto weightTensorNumTiles = getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    auto distributedInput1TensorType = createDistributedInputTensorType(
            origOp, origOp.input1(), activationTensorDistributionMode, activationTensorNumTiles);
    auto distributedInput2TensorType = createDistributedInputTensorType(
            origOp, origOp.input2(), weightsTensorDistributionMode, weightTensorNumTiles);
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);

    return origOp.fitIntoCMX(distributedInput1TensorType, distributedInput2TensorType, distributedOutputTensorType);
}

void StrategyManager::assignMultiClusterStrategy() {
    const auto callback = [this](mlir::Operation* origOp) {
        llvm::TypeSwitch<mlir::Operation*, void>(origOp)
                .Case<NCEMaxPoolOp>([this](NCEMaxPoolOp origOp) {
                    if (_maxPoolStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                        origOp->setAttr(multiClusterStrategy,
                                        mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
                    }
                })
                .Case<NCEEltwiseOp>([this](NCEEltwiseOp origOp) {
                    if (_eltwiseStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                        origOp->setAttr(multiClusterStrategy,
                                        mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
                    }
                })
                .Case<NCEConvolutionOp>([this](NCEConvolutionOp origOp) {
                    // For WW10 channel major convolution will not be excecuted in multi-cluster mode
                    // Only z-major convolution will be considered for multi-cluster mode
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (_convolutionStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                            origOp->setAttr(multiClusterStrategy,
                                            mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
                        }
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (_depthConvolutionStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation())) {
                        origOp->setAttr(multiClusterStrategy,
                                        mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
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
