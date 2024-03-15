//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;
using namespace VPU;

namespace {
SmallVector<Shape> getPerClusterEndOffset(ArrayRef<Shape> startOffset, ArrayRef<Shape> size) {
    const auto numDims = startOffset[0].size();
    const auto numClusters = startOffset.size();
    auto endOffset = SmallVector<Shape>(numClusters, Shape(numDims, 0));

    for (size_t cluster = 0; cluster < numClusters; cluster++) {
        for (size_t dim = 0; dim < numDims; dim++) {
            endOffset[cluster][Dim(dim)] = startOffset[cluster][Dim(dim)] + size[cluster][Dim(dim)] - 1;
        }
    }

    return endOffset;
}

bool isDistributedTensorSOH(VPU::DistributedTensorType distributedTensorType) {
    auto distribution = distributedTensorType.getDistribution();
    const auto mode = distribution.getMode().getValue();

    const bool isSegOverlappedMode =
            mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED;
    if (!isSegOverlappedMode) {
        return false;
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    return numTiles[Dims4D::Act::H.ind()] != 1;
}

bool isValidCandidateForCMXConcat(VPU::ConcatOp concat) {
    bool isProducerOrConsumerSOH = false;
    for (const auto& producerConcat : concat->getOperands()) {
        if (!mlir::isa_and_nonnull<VPU::NCEOpInterface, VPU::NCEClusterTilingOp>(producerConcat.getDefiningOp())) {
            return false;
        }

        if (auto clusterTilingCopy = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(producerConcat.getDefiningOp())) {
            auto copyOp = clusterTilingCopy.getInnerTaskOpOfType<VPU::CopyOp>();
            if (!copyOp) {
                return false;
            }
            auto clusterTilingNCE = clusterTilingCopy->getOperand(0).getDefiningOp<VPU::NCEClusterTilingOp>();
            if (!clusterTilingNCE) {
                return false;
            }
            auto nceOp = clusterTilingNCE.getInnerTaskOpOfType<VPU::NCEOpInterface>();
            if (!nceOp) {
                return false;
            }

            auto distributedTensorType = clusterTilingNCE.getResult(0).getType().dyn_cast<VPU::DistributedTensorType>();
            if (distributedTensorType == nullptr) {
                return false;
            }

            isProducerOrConsumerSOH = isDistributedTensorSOH(distributedTensorType);
        } else if (auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(producerConcat.getDefiningOp())) {
            if (nceOp.getMultiClusterStrategy().has_value()) {
                if (nceOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight) {
                    isProducerOrConsumerSOH = true;
                }
            }
        }
    }
    for (const auto& consumerConcat : concat->getUsers()) {
        if (!mlir::isa_and_nonnull<VPU::NCEOpInterface, VPU::NCEClusterTilingOp>(consumerConcat)) {
            return false;
        }

        if (auto clusterTilingCopy = mlir::dyn_cast<VPU::NCEClusterTilingOp>(consumerConcat)) {
            auto copyOp = clusterTilingCopy.getInnerTaskOpOfType<VPU::CopyOp>();
            if (!copyOp) {
                return false;
            }
            if (!clusterTilingCopy->hasOneUse()) {
                return false;
            }
            auto clusterTilingNCE = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(*clusterTilingCopy->user_begin());
            if (!clusterTilingNCE) {
                return false;
            }
            auto nceOp = clusterTilingNCE.getInnerTaskOpOfType<VPU::NCEOpInterface>();
            if (!nceOp) {
                return false;
            }

            auto distributedTensorType =
                    clusterTilingCopy.getResult(0).getType().dyn_cast<VPU::DistributedTensorType>();
            if (distributedTensorType == nullptr) {
                return false;
            }

            isProducerOrConsumerSOH = isDistributedTensorSOH(distributedTensorType);
        } else if (auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerConcat)) {
            if (nceOp.getMultiClusterStrategy().has_value()) {
                if (nceOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight) {
                    isProducerOrConsumerSOH = true;
                }
            }
        }
    }

    auto isOffsetOnH = [](mlir::ArrayAttr offset) {
        auto offsetVector = Shape(parseIntArrayAttr<int64_t>(offset));
        return offsetVector[Dims4D::Act::H] != 0;
    };

    bool isConcatOverH = false;
    if (concat.getStaticOffsets().has_value()) {
        const auto concatDims = concat.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();
        isConcatOverH = llvm::any_of(concatDims, isOffsetOnH);
    } else if (concat.getPerAxis().has_value()) {
        const auto concatAxis = concat.getPerAxis().value().getAxis().getValue().getSExtValue();
        isConcatOverH = concatAxis == Dims4D::Act::H.ind();
    }

    return !(isProducerOrConsumerSOH && isConcatOverH);
}

int64_t extractKernelTileAxis(ArrayRef<int64_t> numTiles) {
    VPUX_THROW_UNLESS(numTiles[Dims4D::Act::H.ind()] == 1 || numTiles[Dims4D::Act::W.ind()] == 1,
                      "Multidimension cluster tiling across H and W is not yet supported.");
    if (numTiles[Dims4D::Act::W.ind()] > 1) {
        return Dims4D::Kernel::X.ind();
    }
    return Dims4D::Kernel::Y.ind();
}

}  // namespace

OverlapDistributionParams vpux::VPU::getOverlappedDistributionParameters(mlir::MLIRContext* ctx,
                                                                         ArrayRef<VPU::ClusteredOpInterface> opSubgraph,
                                                                         int64_t kernelDistributionAxis,
                                                                         mlir::UnitAttr equalComputeAndMemoryView) {
    auto kernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto pads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    SmallVector<VPU::NCEOpInterface> nceOpCandidates;
    for (auto clusteredOp : opSubgraph) {
        // clusteredOp with SOHO strategy satisfy below SOH condition too so won't be dropped
        if (clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
            if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
                nceOpCandidates.push_back(nceOp);
            }
        }
    }

    if (nceOpCandidates.empty()) {
        return OverlapDistributionParams(kernel, pads, strides);
    }

    // For now just take the highest kernel
    // As we have better representation in distributedBuffer, switch to computing the
    // actual shapes per clusters

    auto largestKernel = 0;
    auto largestIndex = 0;
    for (auto it : nceOpCandidates | indexed) {
        auto kernelSize = it.value().getKernelSizeVal()[kernelDistributionAxis];
        if (kernelSize > largestKernel) {
            largestKernel = kernelSize;
            largestIndex = it.index();
        }
    }

    kernel = getIntArrayAttr(ctx, nceOpCandidates[largestIndex].getKernelSizeVal());
    pads = nceOpCandidates[largestIndex].getPad();
    strides = getIntArrayAttr(ctx, nceOpCandidates[largestIndex].getStridesVal());

    return OverlapDistributionParams(kernel, pads, strides, equalComputeAndMemoryView);
}

OverlapDistributionParams vpux::VPU::getOverlappedDistributionParameters(
        mlir::MLIRContext* ctx, VPU::ClusteredOpInterface producer,
        ArrayRef<VPU::ClusteredOpInterface> consumerSubgraph, const int64_t numClusters, ArrayRef<int64_t> numTiles,
        mlir::UnitAttr uniformDistributedSegments) {
    VPUX_THROW_WHEN(producer == nullptr, "getOverlappedDistributionParameters: producer cannot be nullptr");

    auto numClustersAttr = getIntAttr(ctx, numClusters);
    auto neutralKernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto neutralPads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    auto neutralStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    auto numTilesAttr = getIntArrayAttr(ctx, numTiles);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);
    auto neutralDistributedAttr = VPU::DistributedTensorAttr::get(
            ctx, distributionModeAttr, numTilesAttr, neutralKernel, neutralPads, neutralStrides, numClustersAttr,
            nullptr, uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto producerOutputShape = producer->getResult(0).getType().cast<NDTypeInterface>().getShape();

    auto memoryShapes = getPerClusterComputeShapes(producerOutputShape, neutralDistributedAttr);
    auto memoryOffsets = getPerClusterComputeShapeOffsets(producerOutputShape, neutralDistributedAttr);

    const auto kernelTileAxis = extractKernelTileAxis(numTiles);
    SmallVector<VPU::NCEOpInterface> nceOpCandidates;
    for (auto clusteredOp : consumerSubgraph) {
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
            // clusteredOp with SOHO strategy satisfy below SOH condition too so won't be dropped
            if (kernelTileAxis == Dims4D::Kernel::Y.ind() &&
                clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
                nceOpCandidates.push_back(nceOp);
            }

            if (kernelTileAxis == Dims4D::Kernel::X.ind() &&
                clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                /*axis=*/ShapeRef())) {
                nceOpCandidates.push_back(nceOp);
            }
        }
    }

    if (nceOpCandidates.empty()) {
        return OverlapDistributionParams(vpux::getIntArrayOfArray(ctx, memoryShapes),
                                         vpux::getIntArrayOfArray(ctx, memoryOffsets));
    }

    const auto clusteringAxis = VPU::getDistributedTilingAxis(numTiles);
    for (auto nceOp : nceOpCandidates) {
        auto kernelSize = getIntArrayAttr(ctx, nceOp.getKernelSizeVal());
        auto stridesSize = getIntArrayAttr(ctx, nceOp.getStridesVal());
        auto padsSize = nceOp.getPad();

        auto consumerDistr = VPU::DistributedTensorAttr::get(
                ctx, distributionModeAttr, numTilesAttr, kernelSize, padsSize, stridesSize, numClustersAttr, nullptr,
                uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

        auto consumerMemoryShapesOpt = getPerClusterMemoryShapes(producerOutputShape, consumerDistr);
        VPUX_THROW_WHEN(!consumerMemoryShapesOpt.has_value(),
                        "Wrong kernel, pads, strides for NCEOp. Cannot compute per cluster memory shapes: shape = {0}, "
                        "distribution = {1}",
                        producerOutputShape, consumerDistr);

        auto consumerMemoryShapes = consumerMemoryShapesOpt.value();
        auto consumerMemoryOffsets = getPerClusterMemoryShapeOffsets(producerOutputShape, consumerDistr);

        for (int64_t cluster = 0; cluster < numClusters; ++cluster) {
            auto endOffset =
                    memoryOffsets[cluster][Dim(clusteringAxis)] + memoryShapes[cluster][Dim(clusteringAxis)] - 1;
            const auto candidateEndOffset = consumerMemoryOffsets[cluster][Dim(clusteringAxis)] +
                                            consumerMemoryShapes[cluster][Dim(clusteringAxis)] - 1;

            if (endOffset < candidateEndOffset) {
                endOffset = candidateEndOffset;
            }

            auto startOffset = memoryOffsets[cluster][Dim(clusteringAxis)];
            const auto candidateStartOffset = consumerMemoryOffsets[cluster][Dim(clusteringAxis)];
            if (startOffset > candidateStartOffset) {
                startOffset = candidateStartOffset;
            }

            memoryOffsets[cluster][Dim(clusteringAxis)] = startOffset;
            memoryShapes[cluster][Dim(clusteringAxis)] = endOffset - startOffset + 1;
        }
    }

    return OverlapDistributionParams(vpux::getIntArrayOfArray(ctx, memoryShapes),
                                     vpux::getIntArrayOfArray(ctx, memoryOffsets));
}

OverlapDistributionParams vpux::VPU::getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                                   ArrayRef<int64_t> activationTensorNumTiles) {
    const auto ctx = clusteredOp.getContext();

    const auto kernelTileAxis = extractKernelTileAxis(activationTensorNumTiles);
    const auto localOverlappedParams = getOverlappedDistributionParameters(
            ctx, SmallVector<VPU::ClusteredOpInterface>({clusteredOp}), kernelTileAxis);

    // For 30XX, 37XX, we do not set input workloads explicitly and therefore
    // OVERLAPPED should only represent the current op's input needs w/o
    // the sibling requirements
    return localOverlappedParams;
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                               ArrayRef<int64_t> outputTensorNumTiles,
                                                               vpux::NDTypeInterface outputType) {
    const auto ctx = clusteredOp.getContext();
    SmallVector<VPU::ClusteredOpInterface> consumerSubgraph;
    auto archKind = getArch(clusteredOp.getOperation());
    const std::set<VPU::ArchKind> compatibleTargets = {};
    const auto equalComputeAndMemoryView =
            (compatibleTargets.count(archKind) <= 0 || mlir::isa<NCEPermuteQuantizeOp>(clusteredOp.getOperation()))
                    ? mlir::UnitAttr::get(clusteredOp.getContext())
                    : nullptr;

    if (auto eltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(clusteredOp.getOperation())) {
        if (eltwise.getIsInplace().value_or(false)) {
            auto kernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
            auto pads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
            auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

            return OverlapDistributionParams(kernel, pads, strides);
        }
    }

    for (const auto& result : clusteredOp->getResults()) {
        for (const auto& consumer : result.getUsers()) {
            if (auto clusteredConsumer = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumer)) {
                consumerSubgraph.push_back(clusteredConsumer);
            }

            // Given the following subgraph:
            //     NCEProducer
            //         |
            //       Concat
            //      /    |
            // NceOp0  NceOp1
            // NCEProducer's effective consumers should be: {NceOp0, NceOp1}
            if (auto concatConsumer = mlir::dyn_cast_or_null<VPU::ConcatOp>(consumer)) {
                if (isValidCandidateForCMXConcat(concatConsumer)) {
                    for (const auto& consumerConcat : concatConsumer->getUsers()) {
                        if (auto clusteredConsumerConcat = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerConcat)) {
                            consumerSubgraph.push_back(clusteredConsumerConcat);
                        }
                    }
                }
            }

            // Given the following subgraph:
            //     NCEProducer
            //         |
            //     QuantizeCast
            //         |
            //       NceOp
            // NCEProducer's effective consumer should be NceOp
            // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
            if (auto quantizeCastConsumer = mlir::dyn_cast_or_null<VPU::QuantizeCastOp>(consumer)) {
                for (const auto& consumerQuantizeCast : quantizeCastConsumer->getUsers()) {
                    if (auto clusteredConsumerQuantizeCast =
                                mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerQuantizeCast)) {
                        consumerSubgraph.push_back(clusteredConsumerQuantizeCast);
                    }
                }
            }
        }
    }
    const auto kernelTileAxis = extractKernelTileAxis(outputTensorNumTiles);
    const auto candidateOverlappedParams = getOverlappedDistributionParameters(
            clusteredOp.getContext(), consumerSubgraph, kernelTileAxis, equalComputeAndMemoryView);

    // Lacking a way specifying explicit per cluster shapes and offsets in the distributed
    // datatype, we are forced to pick a configuration where compute view is within the boundaries
    // of the memory view.
    // We represent input workload start & end through compute offset and size, while the total
    // amount of data in cluster is represented through memory shape. In cases where
    // compute start < memory start or compute end > memory_end, the size of data in cluster should be
    // max(compute end, memory_end) - min(compute start, memory start) + 1, but we currently have no way of
    // representing that. Therefore, we ensure that such a case will not happen by setting overlapped params k1x1,
    // s1x1, pad0x0x0x0 if the consumer distribution does not satisfy the requirements.

    const auto numTilesPerDim = (kernelTileAxis == Dims4D::Kernel::Y.ind())
                                        ? outputTensorNumTiles[Dims4D::Act::H.ind()]
                                        : outputTensorNumTiles[Dims4D::Act::W.ind()];
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);

    const auto candidateDistributedAttr = DistributedTensorAttr::get(
            ctx, distributionModeAttr, getIntArrayAttr(ctx, outputTensorNumTiles), candidateOverlappedParams.kernel,
            candidateOverlappedParams.pads, candidateOverlappedParams.stride, getIntAttr(ctx, numTilesPerDim), nullptr,
            mlir::UnitAttr::get(ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto kernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto pads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto fallbackOverlappedParams = OverlapDistributionParams(kernel, pads, strides, equalComputeAndMemoryView);

    const auto outputShape = (outputType == nullptr) ? getShape(clusteredOp->getResult(0)) : outputType.getShape();
    const auto optionalCandidateMemoryShapes = getPerClusterMemoryShapes(outputShape, candidateDistributedAttr);
    if (!optionalCandidateMemoryShapes.has_value()) {
        // If NCEProducer has tiling required and the tiled shape does not satisfy producer op
        return fallbackOverlappedParams;
    }

    const auto candidateMemoryOffsets = getPerClusterMemoryShapeOffsets(outputShape, candidateDistributedAttr);
    const auto candidateComputeOffsets = getPerClusterComputeShapeOffsets(outputShape, candidateDistributedAttr);
    const auto candidateComputeShapes = getPerClusterComputeShapes(outputShape, candidateDistributedAttr);

    // Memory start offset must be before or equal to compute start offset
    for (auto startOffsetsPerClusterZip : zip(candidateMemoryOffsets, candidateComputeOffsets)) {
        for (auto dimZip : zip(std::get<0>(startOffsetsPerClusterZip), std::get<1>(startOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) > std::get<1>(dimZip)) {
                // candidate shape does not satisfy producer op
                return fallbackOverlappedParams;
            }
        }
    }

    const auto candidateMemoryShapes = optionalCandidateMemoryShapes.value();
    const auto candidateMemoryEndOffset = getPerClusterEndOffset(candidateMemoryOffsets, candidateMemoryShapes);
    const auto candidateComputeEndOffset = getPerClusterEndOffset(candidateComputeOffsets, candidateComputeShapes);

    // Memory end offset must be after or equal to compute end offset
    for (auto endOffsetsPerClusterZip : zip(candidateMemoryEndOffset, candidateComputeEndOffset)) {
        for (auto dimZip : zip(std::get<0>(endOffsetsPerClusterZip), std::get<1>(endOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) < std::get<1>(dimZip)) {
                // candidate shape does not satisfy local op
                return fallbackOverlappedParams;
            }
        }
    }

    return candidateOverlappedParams;
}
