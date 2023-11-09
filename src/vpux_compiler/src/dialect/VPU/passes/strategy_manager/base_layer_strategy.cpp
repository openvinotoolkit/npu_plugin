//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

BaseLayerStrategy::BaseLayerStrategy(mlir::func::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceEngine = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    auto dpuExec = nceEngine.getSubExecutor(VPU::ExecutorKind::DPU);
    _numClusters = nceEngine.count();
    _numDPUs = dpuExec.count();
    _minimumOutputHeightForSOH = _numClusters;
    _minimumOutputWidthForSOW = _numClusters;
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                             ShapeRef outputShape) const {
    const auto arch = VPU::getArch(clusteredOp);
    if (outputShape == ShapeRef()) {
        outputShape = getShape(clusteredOp->getResult(0));
    }
    auto isUniformDistributedSegments = !VPU::isArchVPUX3XXX(arch);
    auto heightCompatibleCheck = [&](ShapeRef outputShape) {
        const auto OH = outputShape[Dims4D::Act::H];
        auto numClustersForSOH = getNumberOfClustersForSpatialDim(outputShape[Dims4D::Act::H], _numClusters,
                                                                  isUniformDistributedSegments);
        // Each cluster should be used. When it is just with 3 or 2 clusters, there is an accuracy issue.
        // TODO: Find the root cause for this accuracy regression, E#41297
        auto isSOHCompatible = (OH >= _minimumOutputHeightForSOH && numClustersForSOH == _numClusters);
        return isSOHCompatible;
    };

    auto isSOHCompatible = heightCompatibleCheck(outputShape);

    return isSOHCompatible;
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOW
// compatible it must have an output width of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output Width must be a minimum of 4.
bool BaseLayerStrategy::isOperationSplitOverWidthCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                            ShapeRef outputShape) const {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(clusteredOp->getResult(0));
    }

    auto widthCompatibleCheck = [&](ShapeRef outputShape) {
        const auto OW = outputShape[Dims4D::Act::W];
        auto numClustersForSOW = getNumberOfClustersForSpatialDim(outputShape[Dims4D::Act::W], _numClusters, true);
        // Each cluster should be used. When it is just with 3 or 2 clusters, there is an accuracy issue.
        // TODO: Find the root cause for this accuracy regression, E#41297
        auto isSOWCompatible = (OW >= _minimumOutputWidthForSOW && numClustersForSOW == _numClusters);
        return isSOWCompatible;
    };

    auto isSOWCompatible = widthCompatibleCheck(outputShape);

    return isSOWCompatible;
}

/// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
/// compatible it must have an output channel of at least the number of clusters x 16
/// specified for compilation.
/// For example for 4 cluster compilation the output channel must be a
/// minimum of 4x16=64.
/// @warning Considering SOK can use 2/3 clusters to avoid per cluster channel alignment, like
/// OC = 64, [32, 32] output channels per cluster is valid too.
/// Thus the conditions can be relaxed.
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                             ShapeRef outputShape) const {
    if (outputShape == ShapeRef()) {
        outputShape = getShape(clusteredOp->getResult(0));
    }
    const auto OC = outputShape[Dims4D::Act::C];

    // Sparse Eltwise consuming SOK activations leads to the storage element size different than the number of input
    // channels, which is not a validated scenario
    if (clusteredOp->getResult(0).getType().isa<VPU::SparseTensorType>()) {
        const auto hasEltwiseUser = llvm::any_of(clusteredOp->getResult(0).getUsers(), [](mlir::Operation* userOp) {
            return mlir::isa<VPU::NCEEltwiseOp>(userOp);
        });
        if (hasEltwiseUser) {
            return false;
        }
    }
    // Channel alignment is specific for NCE DPU operations and CMX CONCAT
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());
    auto minChannelSize = (nceOp != nullptr) ? _numChannelAlignment * _numClusters : _numClusters;
    if (OC < minChannelSize) {
        return false;
    }

    if (nceOp == nullptr) {
        return true;
    }

    // SOK will split the weights over output channels. If the weights are sparse, it is necessary to make sure that
    // no split will have only sparse values inside, since that would lead to zero-sized weights
    auto weights = nceOp.getWeightsOperand();
    if (weights != nullptr && weights.getType().isa<VPU::SparseTensorType>()) {
        if (const auto compressionScheme = weights.getType().cast<VPU::SparseTensorType>().getCompressionScheme()) {
            // Create a new type with the new number of output channels
            // If the element type is quantized per-axis, it is replaced with a per-tensor type to avoid the
            // incompatibility between the number of elements per axis and the number of scales & zero-points
            const auto origType = weights.getType().cast<vpux::NDTypeInterface>();
            auto newShape = Shape(origType.getShape().raw());
            newShape[Dims4D::Filter::OC] = OC;
            auto elemType = origType.getElementType();
            if (auto qElemType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
                elemType = mlir::quant::UniformQuantizedType::get(
                        qElemType.getFlags(), qElemType.getStorageType(), qElemType.getExpressedType(), /*scale=*/1.0,
                        /*zeroPoint=*/0, qElemType.getStorageTypeMin(), qElemType.getStorageTypeMax());
            }
            const auto newType = origType.changeShapeElemType(newShape, elemType);

            // Create a distributed type in order to determine the channel split over clusters
            const auto numClustersAttr = getIntAttr(clusteredOp.getContext(), _numClusters);
            const auto filterType = VPU::getDistributedFilterTypeFromOp(nceOp, newType, numClustersAttr,
                                                                        VPU::MultiClusterStrategy::SplitOverKernel);
            const auto filterDistType = filterType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
            const auto computeOffsets = filterDistType.getPerClusterComputeShapeOffsets();
            if (!computeOffsets.empty()) {
                int64_t startOC = computeOffsets[0][Dims4D::Filter::OC];
                for (size_t i = 1; i < computeOffsets.size(); ++i) {
                    const int64_t sizeOC = computeOffsets[i][Dims4D::Filter::OC] - startOC;
                    const auto numElems = compressionScheme.getNumElemsInRange(startOC, sizeOC);
                    if (numElems == 0) {
                        return false;
                    }
                    startOC += sizeOC;
                }
                const auto remainingOC = OC - startOC;
                const auto numElems = compressionScheme.getNumElemsInRange(startOC, remainingOC);
                if (numElems == 0) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool BaseLayerStrategy::doesLayerFitIntoCMX(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                            Byte reservedMem) const {
    return VPU::doesLayerFitIntoCMX(clusteredOp, strategy, reservedMem, _log);
}

bool BaseLayerStrategy::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
        VPU::DistributedTypeInterface newDistributedTensorType) const {
    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(clusteredOp->getName());

    return llvm::TypeSwitch<mlir::Operation*, bool>(clusteredOp.getOperation())
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
            .Case<NCECompressConvolutionOp>([&](NCECompressConvolutionOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(
                        distributedTensorTypes.size() == 3,
                        "VPU::NCECompressConvolutionOp operation should have 3 DistributedTensorType, but got {0}",
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
            .Case<NCEInterpolateOp>([&](NCEInterpolateOp origOp) {
                auto distributedTensorTypes = layerStrategyChecker->getDistributedTensorType(origOp, strategy);
                VPUX_THROW_UNLESS(distributedTensorTypes.size() == 3,
                                  "VPU::NCEInterpolateOp operation should have 3 DistributedTensorType, but got {0}",
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
