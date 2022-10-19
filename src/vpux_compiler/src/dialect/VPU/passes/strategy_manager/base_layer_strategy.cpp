//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

BaseLayerStrategy::BaseLayerStrategy(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    auto dpuExec = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));
    _numClusters = nceOp.count();
    _numDPUs = dpuExec.count();
    _minimumOutputHeightForSOH = _numClusters;
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];
    auto numClustersForSOH = getNumberOfClustersForSOH(outputShape[Dims4D::Act::H], _numClusters);
    // Each cluster should be used. When it is just with 3 or 2 clusters, there is an accuracy issue.
    // TODO: Find the root cause for this accuracy regression, E#41297
    return OH >= _minimumOutputHeightForSOH && numClustersForSOH == _numClusters;
}

/// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
/// compitable it must have an output channel of at least the number of clusters x 16
/// specified for compilation.
/// For example for 4 cluster compilation the output channel must be a
/// minimum of 4x16=64.
/// @warning Considering SOK can use 2/3 clusters to avoid clusters channel alignment, like
/// OC = 48, [16, 16, 16] output channels per cluster is valid too.
/// Thus the conditions can be relaxed.
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(VPU::NCEOpInterface nceOp) const {
    const auto outputShape = getShape(nceOp->getResult(0));
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _numChannelAlignment * 2;
}

bool BaseLayerStrategy::isOperationMultiClusterCompatible(VPU::NCEOpInterface nceOp) const {
    return isOperationSplitOverHeightCompatible(nceOp) || isOperationSplitOverKernelCompatible(nceOp);
}

bool BaseLayerStrategy::doesLayerFitIntoCMX(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(nceOp->getName());
    return llvm::TypeSwitch<mlir::Operation*, bool>(nceOp.getOperation())
            .Case<NCEMaxPoolOp>([&](NCEMaxPoolOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 2,
                                  "VPU::NCEMaxPoolOp operation should have 2 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1]);
            })
            .Case<NCEAveragePoolOp>([&](NCEAveragePoolOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 2,
                                  "VPU::NCEAveragePoolOp operation should have 2 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1]);
            })
            .Case<NCEEltwiseOp>([&](NCEEltwiseOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 3,
                                  "VPU::NCEEltwiseOp operation should have 3 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1],
                                         distributedTensorTypes[2]);
            })
            .Case<NCEConvolutionOp>([&](NCEConvolutionOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 3,
                                  "VPU::NCEConvolutionOp operation should have 3 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1],
                                         distributedTensorTypes[2]);
            })
            .Case<NCEDepthConvolutionOp>([&](NCEDepthConvolutionOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(
                        distributedTensorTypes.size() == 3,
                        "VPU::NCEDepthConvolutionOp operation should have 3 DistributedTensorType, but got {0}",
                        distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1],
                                         distributedTensorTypes[2]);
            })
            .Default([&](mlir::Operation* unknownOp) -> bool {
                _log.trace("Operation '{0}' at '{1}' is not supported by the NCE", unknownOp->getName(),
                           unknownOp->getLoc());
                return false;
            });
}

bool BaseLayerStrategy::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
        VPU::DistributedTensorType newDistributedTensorType) const {
    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(nceOp->getName());

    return llvm::TypeSwitch<mlir::Operation*, bool>(nceOp.getOperation())
            .Case<NCEMaxPoolOp>([&](NCEMaxPoolOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 2,
                                  "VPU::NCEMaxPoolOp operation should have 2 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], newDistributedTensorType);
            })
            .Case<NCEAveragePoolOp>([&](NCEAveragePoolOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 2,
                                  "VPU::NCEAveragePoolOp operation should have 2 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], newDistributedTensorType);
            })
            .Case<NCEConvolutionOp>([&](NCEConvolutionOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 3,
                                  "VPU::NCEConvolutionOp operation should have 3 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1],
                                         newDistributedTensorType);
            })
            .Case<NCEDepthConvolutionOp>([&](NCEDepthConvolutionOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(
                        distributedTensorTypes.size() == 3,
                        "VPU::NCEDepthConvolutionOp operation should have 3 DistributedTensorType, but got {0}",
                        distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1],
                                         newDistributedTensorType);
            })
            .Default([&](mlir::Operation* unknownOp) -> bool {
                _log.trace("Operation '{0}' at '{1}' is not supported change output alignment", unknownOp->getName(),
                           unknownOp->getLoc());
                return false;
            });
}
