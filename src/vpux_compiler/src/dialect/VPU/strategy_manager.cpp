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
// i.e the OH is > 20 and the layer fits in CMX when multi-clustered

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);

    _numClusters = nceOp.count();
}

void StrategyManager::assignMultiClusterStrategy() {
    const auto callback = [this](mlir::Operation* origOp) {
        llvm::TypeSwitch<mlir::Operation*, void>(origOp)
                .Case<NCEMaxPoolOp>([this](NCEMaxPoolOp origOp) {
                    if (isOperationSplitOverHeightCompatible<NCEMaxPoolOp>(origOp) &&
                        doesSplitOverHeightLayerFitIntoCMX<NCEMaxPoolOp>(origOp)) {
                        origOp->setAttr(multiClusterStrategy,
                                        mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
                    }
                })
                .Case<NCEEltwiseOp>([this](NCEEltwiseOp origOp) {
                    if (isOperationSplitOverHeightCompatible<NCEEltwiseOp>(origOp) &&
                        doesSplitOverHeightLayerFitIntoCMX<NCEEltwiseOp>(origOp)) {
                        origOp->setAttr(multiClusterStrategy,
                                        mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
                    }
                })
                .Case<NCEConvolutionOp>([this](NCEConvolutionOp origOp) {
                    // For WW10 channel major convolution will not be excecuted in multi-cluster mode
                    // Only z-major convolution will be considered for multi-cluster mode
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (isOperationSplitOverHeightCompatible<NCEConvolutionOp>(origOp) &&
                            doesSplitOverHeightLayerFitIntoCMX<NCEConvolutionOp>(origOp)) {
                            origOp->setAttr(multiClusterStrategy,
                                            mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
                        }
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (isOperationSplitOverHeightCompatible<NCEDepthConvolutionOp>(origOp) &&
                        doesSplitOverHeightLayerFitIntoCMX<NCEDepthConvolutionOp>(origOp)) {
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
