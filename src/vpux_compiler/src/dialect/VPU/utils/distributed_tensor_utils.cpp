//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"

using namespace vpux;
using namespace VPU;

//
// Distributed tensor utilities
//

// This method computes the number of clusters to be used for an individual SOK
// layer such that additional alignment of the per cluster output channels is not required.
// Example: For 80 output channel / 4 clusters = [20, 20, 20, 20] output channels per cluster.
// 20 is not aligned to 16. Therefore, the compiler should only execute this layer on 3 clusters.
// This would result in [32, 32, 16] output channels per cluster.
int64_t vpux::VPU::getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersToUseForLayer) {
    for (int64_t clusters = numClustersToUseForLayer; clusters >= 1; clusters--) {
        auto alignedOutputChannels = alignVal<int64_t>(divUp(outputChannels, clusters), KMB_DPU_CHANNELS_ALIGNMENT);
        int64_t remainder = outputChannels - (clusters - 1) * alignedOutputChannels;
        if (remainder > 0) {
            return clusters;
        }
    }
    return 1;
}

int64_t vpux::VPU::getNumberOfClustersForSOH(int64_t outputHeight, int64_t numClustersForCompilation) {
    for (int64_t clusters = numClustersForCompilation; clusters >= 1; clusters--) {
        auto alignedOutputHeight = divUp(outputHeight, clusters);
        int64_t remainder = outputHeight - (clusters - 1) * alignedOutputHeight;
        if (remainder > 0) {
            return clusters;
        }
    }
    return 1;
}

SmallVector<int64_t> vpux::VPU::getActivationTensorNumTiles(int64_t numClustersAvailableForCompilation,
                                                            VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
               strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getActivationTensorAlignment(VPU::NCEOpInterface nceOp,
                                                                       VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering) {
        return SmallVector<int64_t>{1, 16, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::HKSwitch) {
        if (auto origOp = mlir::dyn_cast<NCEConvolutionOp>(nceOp.getOperation())) {
            const auto outShape = getShape(origOp.output());
            vpux::TileInfo outputTile{outShape};
            const auto inputTileShape = origOp.backInferTileInfo(outputTile, Logger::global());
            const auto heightAlignment = getSOHPerClusterHeightAlignment(inputTileShape.tiles[0].shape[Dims4D::Act::W]);
            if (heightAlignment <= 1) {
                return None;
            }
            return SmallVector<int64_t>{1, 1, heightAlignment, 1};
        }
    }

    return None;
}

SmallVector<int64_t> vpux::VPU::getOutputTensorNumTiles(vpux::NDTypeInterface tensorType,
                                                        int64_t numClustersAvailableForCompilation,
                                                        VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto OC = tensorType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer =
                getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {1, numClustersToUseForLayer, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "output tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getOutputTensorAlignment(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return SmallVector<int64_t>{1, 16, 1, 1};
    }

    return None;
}

Optional<DistributedTensorType> vpux::VPU::adjustOutputAlignmentForSOH(VPU::NCEOpInterface nceOp,
                                                                       VPU::DistributedTensorType originalDistType) {
    if (nceOp->getResult(0).use_empty()) {
        return None;
    }

    // optimization SOH -> SOH alignment to remove spilling
    // For multi-users just random choose one NCEOp for optimize
    // TODO: choose the best NCEOp or find least common multiple of all user's alignment
    for (auto consumerOp : nceOp->getResult(0).getUsers()) {
        if (!consumerOp->hasAttr(multiClusterStrategy)) {
            continue;
        }

        const auto strategy =
                consumerOp->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();
        if (strategy != VPU::MultiClusterStrategy::SplitOverHeight && strategy != VPU::MultiClusterStrategy::HKSwitch) {
            continue;
        }

        if (auto convOp = mlir::dyn_cast<NCEConvolutionOp>(consumerOp)) {
            const auto arch = VPU::getArch(consumerOp);
            if (VPU::NCEInvariant::isChannelMajorCompatible(arch,
                                                            convOp.input().getType().cast<vpux::NDTypeInterface>())) {
                return None;
            }
        }

        if (auto consumerNCEOp = mlir::dyn_cast<VPU::NCEOpInterface>(consumerOp)) {
            const auto newAlignment =
                    getActivationTensorAlignment(consumerNCEOp, VPU::MultiClusterStrategy::SplitOverHeight);

            if (!newAlignment.hasValue()) {
                return None;
            }

            auto getNewDistributedTensorType = [&](ArrayRef<int64_t> alignment) -> DistributedTensorType {
                const auto newAlignmentAttr = getIntArrayAttr(nceOp->getContext(), alignment);
                auto distributedAttr = originalDistType.getDistribution();
                auto newDistributedAttr = VPU::DistributedTensorAttr::get(
                        distributedAttr.mode(), distributedAttr.num_tiles(), distributedAttr.kernel(),
                        distributedAttr.pads(), distributedAttr.strides(), distributedAttr.num_clusters(),
                        newAlignmentAttr, nceOp->getContext());
                return VPU::DistributedTensorType::get(nceOp->getContext(), originalDistType.getShape().raw(),
                                                       originalDistType.getElementType(), originalDistType.getOrder(),
                                                       originalDistType.getMemSpace(), newDistributedAttr);
            };

            const auto heightAlignment = newAlignment.getValue()[Dims4D::Act::H.ind()];
            const auto perClusterShape = originalDistType.getPerClusterComputeShapes();
            const auto kernelSize = nceOp.getKernelSize();
            if (std::all_of(perClusterShape.begin(), perClusterShape.end() - 1,
                            [&](vpux::ShapeRef shape) {
                                return (shape[Dims4D::Act::H] % heightAlignment == 0);
                            }) ||
                kernelSize[Dims4D::Kernel::Y.ind()] > 1) {
                auto newOutputDistributedTensorType = getNewDistributedTensorType(newAlignment.getValue());

                auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(nceOp->getName());
                if (layerStrategyChecker->doesLayerChangeOutputAlignmentFitIntoCMX(
                            nceOp, VPU::MultiClusterStrategy::SplitOverHeight, newOutputDistributedTensorType)) {
                    return newOutputDistributedTensorType;
                }
            }
        }
    }

    return None;
}

SmallVector<int64_t> vpux::VPU::getWeightsTensorNumTiles(vpux::NDTypeInterface tensorType,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto OC = tensorType.getShape()[Dims4D::Filter::OC];
        int64_t numClustersToUseForLayer =
                getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

Optional<SmallVector<int64_t>> vpux::VPU::getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering) {
        return SmallVector<int64_t>{16, 1, 1, 1};
    }
    return None;
}

SmallVector<int64_t> vpux::VPU::getWeightsTableTensorNumTiles(vpux::NDTypeInterface tensorType,
                                                              int64_t numClustersAvailableForCompilation,
                                                              VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto OC = tensorType.getShape()[Dims4D::Act::C];
        int64_t numClustersToUseForLayer =
                getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersAvailableForCompilation);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

SmallVector<int64_t> vpux::VPU::getActivationWindowTensorNumTiles(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, 1, 1};
    }
    VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
               "activation window tensor",
               strategy);
}

SmallVector<int64_t> vpux::VPU::getInstructionListTableTensorNumTiles(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, 1, 1};
    }
    VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
               "instruction list table tensor",
               strategy);
}

DistributionMode vpux::VPU::getActivationTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
               strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getWeightsTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "weights tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getOutputTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::DUPLICATED | DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return DistributionMode::MULTICASTED | DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "output tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getActivationWindowTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   " activation window tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getInstructionListTableTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return DistributionMode::DUPLICATED;
    }

    VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
               "instruction list table tensor",
               strategy);
}

NCEClusterTilingOp vpux::VPU::createDistributedCopyOut(VPU::NCEOpInterface nceOp, NCEClusterTilingOp clusterTilingOp) {
    mlir::OpBuilder builder(nceOp);
    auto origOutput = nceOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    return builder.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0),
                                              outputTensorBodyBuilder);
}

NCEClusterTilingOp vpux::VPU::createDistributedCopyOut(mlir::Operation* sourceOp, vpux::NDTypeInterface outputType) {
    mlir::OpBuilder builder(sourceOp);
    builder.setInsertionPointAfter(sourceOp);
    const auto origOutMemSpace = IndexedSymbolAttr::get(sourceOp->getContext(), stringifyEnum(MemoryKind::DDR), 0);

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    return builder.create<NCEClusterTilingOp>(sourceOp->getLoc(), outputType, sourceOp->getResult(0),
                                              outputTensorBodyBuilder);
}

int64_t vpux::VPU::getSOHPerClusterHeightAlignment(int64_t inputWidth) {
    // W * h_per_cluster must be divisible by 4, thus
    // if W % 4 == 0, then h alignment needs to be 1
    // if W % 2 == 0, then h alignment needs to be 2
    // else h alignment needs to be 4
    if (inputWidth % 4 == 0) {
        return 1;
    } else if (inputWidth % 2 == 0) {
        return 2;
    }

    return 4;
}

// When doing SOH not all combinations are supported by HW in terms of how input is segmented
// Following rules need to be satisfied:
// - height of clusters from 0 to N - 1 must be equal
// - height of last cluster (which stores the remainder) must be <= of height of previous clusters
// Additional requirement if operation is not of depth-wise type (it is needed for CONV but not for DWCONV or
// MAXPOOL)
// - Width * height_per_cluster (for cluster 0 - N-1) must be multiple of 4
bool vpux::VPU::isSOHSupportedByDPU(ShapeRef inputShape, int64_t numClusters, bool DWTypeOp) {
    const auto IH = inputShape[Dims4D::Act::H];
    const auto IW = inputShape[Dims4D::Act::W];

    auto hPerCluster = divUp(IH, numClusters);
    auto alignment = (DWTypeOp ? 1 : getSOHPerClusterHeightAlignment(IW));

    hPerCluster = alignVal(hPerCluster, alignment);

    auto hLastCluster = IH - hPerCluster * (numClusters - 1);

    return (hLastCluster > 0);
}

mlir::IntegerAttr vpux::VPU::getOptimalNumClusters(VPU::NCEOpInterface nceOp, int64_t OC,
                                                   VPU::MultiClusterStrategy strategy) {
    auto* ctx = nceOp->getContext();
    auto module = nceOp->getParentOfType<mlir::ModuleOp>();
    auto nceResOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClustersAvailableForCompilation = getIntAttr(ctx, nceResOp.count());
    auto optimalNumberOfClusters = numClustersAvailableForCompilation;

    // Here the number of clusters to be used for an individual SOK layer is determined
    // such that additional alignment of the per cluster output channels is not required.
    // For example 80 output channels, the weights should only be split on 3 clusters [32, 32, 16].
    // Also when creating the copy-in for the activation we need to ensure that the number
    // of clusters that the input is duplicated to is also 3 clusters in this case.
    // Therefore we use the variable optimalNumberOfClusters for both purposes here, to detemine
    // num_tiles and numClusters for the activations and the weights.
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        int64_t numClustersToUseForLayer = numClustersAvailableForCompilation.getValue().getSExtValue();
        numClustersToUseForLayer = getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersToUseForLayer);
        optimalNumberOfClusters = mlir::IntegerAttr::get(getInt64Type(ctx), numClustersToUseForLayer);
    }
    return optimalNumberOfClusters;
}

DistributedTensorType vpux::VPU::createDistributedTensorType(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             DistributionMode distributionMode,
                                                             mlir::ArrayAttr numTiles,
                                                             mlir::IntegerAttr optimalNumberOfClusters,
                                                             mlir::ArrayAttr alignment) {
    auto* ctx = nceOp->getContext();
    DistributedTensorAttr distributedActivationTensorAttr;
    auto module = nceOp->getParentOfType<mlir::ModuleOp>();
    auto nceResOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(ctx, distributionMode);

    auto kernel = getIntArrayAttr(ctx, nceOp.getKernelSize());
    const auto shape = inputType.getShape();
    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto stride = getIntArrayAttr(ctx, nceOp.getStrides());
        auto pad = nceOp.getPad();
        optimalNumberOfClusters = getIntAttr(ctx, nceResOp.count());

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           optimalNumberOfClusters, alignment, ctx);
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, ctx);
    } else if (VPU ::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, ctx);

    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
    }

    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(ctx, MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
    auto elemType = inputType.getElementType();

    return DistributedTensorType::get(ctx, shape.raw(), elemType, order, memSpace, distributedActivationTensorAttr);
}

NCEClusterTilingOp vpux::VPU::createDistributedCopyIn(VPU::NCEOpInterface nceOp, mlir::Value input,
                                                      DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                      mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy) {
    auto OC = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    auto numClusters = getOptimalNumClusters(nceOp, OC, strategy);
    auto inputTensorDistributedTensorType = createDistributedTensorType(
            nceOp, input.getType().cast<vpux::NDTypeInterface>(), distributionMode, numTiles, numClusters, alignment);

    mlir::OpBuilder builder(nceOp);
    builder.setInsertionPoint(nceOp);
    const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN));
        auto inputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(nceOp->getLoc(), newOperands[0], memSpace);
        builder.create<YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
    };

    auto distributedInputCopyOp = builder.create<NCEClusterTilingOp>(nceOp->getLoc(), inputTensorDistributedTensorType,
                                                                     input, inputTensorBodyBuilder);

    return distributedInputCopyOp;
}

inline bool needToAlign(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType) {
    // No need to align CMajor NCE Conv ops
    // Eltwise operations do not need to align but the "alignment" attribute is required
    // to keep the continuity of the distribution type
    return !(mlir::isa<VPU::NCEConvolutionOp>(nceOp) &&
             VPU::NCEInvariant::isChannelMajorCompatible(VPU::getArch(nceOp.getOperation()), inputType));
}

DistributedTensorType vpux::VPU::getDistributedActivationTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                    vpux::NDTypeInterface inputType,
                                                                    mlir::IntegerAttr numClusters) {
    VPUX_THROW_UNLESS(nceOp->hasAttr(multiClusterStrategy), "Op {0} does not have multiClusterStrategy attribute",
                      nceOp->getLoc());
    return getDistributedActivationTypeFromOp(
            nceOp, inputType, numClusters,
            nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
}

DistributedTensorType vpux::VPU::getDistributedActivationTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                    vpux::NDTypeInterface inputType,
                                                                    mlir::IntegerAttr numClusters,
                                                                    VPU::MultiClusterStrategy customStrategy) {
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(customStrategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(nceOp.getContext(), getActivationTensorNumTiles(numClusters.getInt(), customStrategy));

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    if (needToAlign(nceOp, inputType)) {
        const auto activationAlignment = getActivationTensorAlignment(nceOp, customStrategy);
        if (activationAlignment.hasValue()) {
            activationAlignmentAttr = getIntArrayAttr(nceOp.getContext(), activationAlignment.getValue());
        }
    }

    return createDistributedTensorType(nceOp, inputType, activationTensorDistributionMode, activationTensorNumTiles,
                                       numClusters, activationAlignmentAttr);
}

DistributedTensorType vpux::VPU::getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                vpux::NDTypeInterface inputType,
                                                                mlir::IntegerAttr numClusters) {
    VPUX_THROW_UNLESS(nceOp->hasAttr(multiClusterStrategy), "Op {0} does not have multiClusterStrategy attribute",
                      nceOp->getLoc());
    return getDistributedFilterTypeFromOp(
            nceOp, inputType, numClusters,
            nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
}

DistributedTensorType vpux::VPU::getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                vpux::NDTypeInterface inputType,
                                                                mlir::IntegerAttr numClusters,
                                                                VPU::MultiClusterStrategy customStrategy) {
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(customStrategy);
    const auto weightsTensorNumTiles = getIntArrayAttr(
            nceOp.getContext(), getWeightsTensorNumTiles(inputType, numClusters.getInt(), customStrategy));

    const auto weightAlignment = getWeightsTensorAlignment(customStrategy);

    if (weightAlignment.hasValue()) {
        weightAlignmentAttr = getIntArrayAttr(nceOp.getContext(), weightAlignment.getValue());
    }

    return createDistributedTensorType(nceOp, inputType, weightsTensorDistributionMode, weightsTensorNumTiles,
                                       numClusters, weightAlignmentAttr);
}

DistributedTensorType vpux::VPU::getDistributedOutputTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                vpux::NDTypeInterface inputType,
                                                                mlir::IntegerAttr numClusters) {
    VPUX_THROW_UNLESS(nceOp->hasAttr(multiClusterStrategy), "Op {0} does not have multiClusterStrategy attribute",
                      nceOp->getLoc());
    return getDistributedOutputTypeFromOp(
            nceOp, inputType, numClusters,
            nceOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
}

DistributedTensorType vpux::VPU::getDistributedOutputTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                vpux::NDTypeInterface inputType,
                                                                mlir::IntegerAttr numClusters,
                                                                VPU::MultiClusterStrategy customStrategy) {
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(customStrategy);
    const auto outputTensorNumTiles = getIntArrayAttr(
            nceOp.getContext(), getOutputTensorNumTiles(inputType, numClusters.getInt(), customStrategy));

    mlir::ArrayAttr outputAlignmentAttr = nullptr;
    if (needToAlign(nceOp, nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>())) {
        const auto outputAlignment = getOutputTensorAlignment(customStrategy);
        if (outputAlignment.hasValue()) {
            outputAlignmentAttr = getIntArrayAttr(nceOp.getContext(), outputAlignment.getValue());
        }
    }

    return createDistributedTensorType(nceOp, inputType, outputTensorDistributionMode, outputTensorNumTiles,
                                       numClusters, outputAlignmentAttr);
}

Shape vpux::VPU::getLargestClusterOutputShape(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) {
    auto outputType = nceOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    const int64_t OC = outputType.getShape()[Dims4D::Act::C];
    auto numClustersAttr = getOptimalNumClusters(nceOp, OC, strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputType, numClustersAttr, strategy);
    return distributedOutputTensorType.getLargestCompactShape();
}

SmallVector<Shape> vpux::VPU::getPerClusterOutputShape(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) {
    auto outputType = nceOp->getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();
    const int64_t OC = outputType.getShape()[Dims4D::Act::C];
    auto numClustersAttr = getOptimalNumClusters(nceOp, OC, strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(nceOp, outputType, numClustersAttr, strategy);
    return distributedOutputTensorType.getPerClusterComputeShapes();
}
