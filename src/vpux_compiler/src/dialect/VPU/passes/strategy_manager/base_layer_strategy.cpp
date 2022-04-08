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
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface nceOp,
                                                             ShapeRef outputShape) const {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(nceOp->getResult(0));
    }
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
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface nceOp,
                                                             ShapeRef outputShape) const {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(nceOp->getResult(0));
    }
    const auto OC = outputShape[Dims4D::Act::C];

    // Operations are not compatible with SOK when producing sparse activations, if these activations are not
    // split equally across clusters based on the number of channels.
    // The consumer operation of the sparse activation must then treat each section of channels separately
    // for desparsification, which requires them to be equal in number due to the storage element size register.
    if (nceOp->getResult(0).getType().isa<VPU::SparseTensorType>()) {
        auto moduleOp = nceOp->getParentOfType<mlir::ModuleOp>();
        auto nceResOp = IE::getAvailableExecutor(moduleOp, ExecutorKind::NCE);
        const auto numClusters = nceResOp.count();

        auto perClusterOC = divUp(OC, numClusters);
        perClusterOC = alignVal<int64_t>(perClusterOC, _numChannelAlignment);
        if (perClusterOC * numClusters != OC) {
            return false;
        }

        // Eltwise consuming SOK activations leads to the storage element size different than the number of input
        // channels, which is not a validated scenario
        const auto hasEltwiseUser = llvm::any_of(nceOp->getResult(0).getUsers(), [](mlir::Operation* userOp) {
            return mlir::isa<VPU::NCEEltwiseOp>(userOp);
        });
        if (hasEltwiseUser) {
            return false;
        }
    }

    return OC >= _numChannelAlignment * 2;
}

bool BaseLayerStrategy::doesLayerFitIntoCMX(VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
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
            .Case<SWOpInterface>([&](SWOpInterface origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(nceOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 2,
                                  "VPU::SWOpInterface operation should have 2 DistributedTensorType, but got {0}",
                                  distributedTensorTypes.size());
                return origOp.fitIntoCMX(distributedTensorTypes[0], distributedTensorTypes[1]);
            })
            .Default([&](mlir::Operation* unknownOp) -> bool {
                _log.trace("Operation '{0}' at '{1}' is not supported by the NCE", unknownOp->getName(),
                           unknownOp->getLoc());
                return false;
            });
}

bool BaseLayerStrategy::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy,
        VPU::DistributedTypeInterface newDistributedTensorType) const {
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
