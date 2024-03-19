//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;

VPU::DistributedTensorAttr VPU::getSWExplicitDistributedTensorAttr(
        VPU::SWOpInterface swOp, ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments) {
    VPUX_THROW_WHEN(swOp == nullptr, "Cannot get SW DistributedTensorAttr, is not a SW op");
    auto ctx = swOp.getContext();
    const auto actTensorDistrModeAttr = VPU::DistributionModeAttr::get(ctx, distributionMode);

    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        const auto outShape = getShape(swOp->getResult(0));
        std::optional<ArrayRef<int64_t>> alignmentValue =
                alignment == nullptr ? std::nullopt
                                     : std::optional<ArrayRef<int64_t>>(parseIntArrayAttr<int64_t>(alignment));

        const auto tiles = fillDividedTiles(Shape(parseIntArrayAttr<int64_t>(numTiles)), outShape, alignmentValue);
        VPUX_THROW_WHEN(mlir::failed(tiles), "Incorrect tiles at {0}", swOp.getLoc());
        const auto outTiles = tiles.value();

        auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilder == nullptr, "Cannot cast op to TilingBuilderOpInterface at {0}", swOp.getLoc());
        SmallVector<InputTiling> inputTiles;
        for (const auto& outTile : outTiles) {
            inputTiles.push_back(tilingBuilder.backInferTileInfo(outTile, Logger::global()));
            VPUX_THROW_UNLESS(inputTiles.back().tiles.size() == 1, "Unexpected input operands size {0}",
                              inputTiles.back().tiles.size());
        }

        SmallVector<SmallVector<int64_t>> inputPerClusterShape;
        SmallVector<SmallVector<int64_t>> inputPerClusterOffset;
        for (auto i : irange(outTiles.size())) {
            inputPerClusterShape.push_back(to_small_vector(inputTiles[i].tiles.front().shape));
            inputPerClusterOffset.push_back(to_small_vector(inputTiles[i].tiles.front().offsets));
        }

        auto perClusterShapeAttr = vpux::getIntArrayOfArray(ctx, inputPerClusterShape);
        auto perClusterOffsetAttr = vpux::getIntArrayOfArray(ctx, inputPerClusterOffset);
        return VPU::DistributedTensorAttr::get(ctx, actTensorDistrModeAttr, numTiles, nullptr, nullptr, nullptr,
                                               numClusters, alignment, uniformDistributedSegments, perClusterShapeAttr,
                                               perClusterOffsetAttr, perClusterShapeAttr, perClusterOffsetAttr,
                                               nullptr);
    }

    return getNonOverlappedDistributedAttr(shape, actTensorDistrModeAttr, numTiles, numClusters, alignment,
                                           uniformDistributedSegments, ctx);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::getNCEExplicitDistributedTensorAttr(
        VPU::NCEOpInterface nceOp, ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, PaddingAttr pad,
        mlir::ArrayAttr stride, mlir::UnitAttr uniformDistributedSegments) {
    VPUX_THROW_WHEN(nceOp == nullptr, "Cannot get HW DistributedTensorAttr, is not a HW op");
    auto ctx = nceOp.getContext();

    // For a SOK layer with sparse output, try not using uniformDistributedSegments because NCE operations with sparse
    // outputs must have all variants with the same number of channels excluding the last one
    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        const auto numTilesArr = parseIntArrayAttr<int64_t>(numTiles);
        const auto axis = vpux::VPU::getDistributedTilingAxis(numTilesArr);
        if ((nceOp->getResult(0).getType().isa<VPU::SparseTensorType>()) &&
            (axis == Dims4D::Act::C.ind() || axis == Dims4D::Filter::OC.ind())) {
            auto rawShape = to_small_vector(shape.raw());
            auto tiledShape = rawShape;
            auto remainderTileShape = rawShape;
            // Split in an equal manner such that first N-1 tiles are equal
            // and the last tile can be less or equal.
            tiledShape[axis] = divUp(tiledShape[axis], numTilesArr[axis]);

            if (alignment != nullptr) {
                const auto alignmentArray = parseIntArrayAttr<int64_t>(alignment);
                tiledShape = alignShape(tiledShape, alignmentArray, alignValUp<int64_t>);
            }

            // Last tile will have the remainder and it doesn't have to be aligned
            remainderTileShape[axis] = rawShape[axis] - tiledShape[axis] * (numTilesArr[axis] - 1);
            if (remainderTileShape[axis] > 0) {
                uniformDistributedSegments = nullptr;
            }
        }
    }

    const auto actTensorDistrModeAttr = VPU::DistributionModeAttr::get(ctx, distributionMode);
    VPU::DistributedTensorAttr distributedActivationTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, actTensorDistrModeAttr, numTiles, kernel, pad, stride, numClusters, alignment,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto perClusterComputeShapes =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapes(shape, distributedActivationTensorAttr));
    auto perClusterComputeOffsets = vpux::getIntArrayOfArray(
            ctx, VPU::getPerClusterComputeShapeOffsets(shape, distributedActivationTensorAttr));
    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedActivationTensorAttr);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}",
                      distributedActivationTensorAttr);
    auto perClusterMemoryShapes = vpux::getIntArrayOfArray(ctx, optionalClusterMemoryShapes.value());
    auto perClusterMemoryOffsets =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(shape, distributedActivationTensorAttr));

    return VPU::DistributedTensorAttr::get(ctx, actTensorDistrModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           numClusters, alignment, uniformDistributedSegments, perClusterComputeShapes,
                                           perClusterComputeOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                                           nullptr);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::getConcatExplicitDistributedAttr(
        ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
        mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, VPU::PaddingAttr pad, mlir::ArrayAttr stride,
        mlir::UnitAttr uniformDistributedSegments, mlir::MLIRContext* ctx) {
    const auto actTensorDistrModeAttr = VPU::DistributionModeAttr::get(ctx, distributionMode);
    auto distributedActivationTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, actTensorDistrModeAttr, numTiles, kernel, pad, stride, numClusters, alignment,
            uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedActivationTensorAttr);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}",
                      distributedActivationTensorAttr);
    auto perClusterMemoryShapes = getIntArrayOfArray(ctx, optionalClusterMemoryShapes.value());
    auto perClusterMemoryOffsets =
            getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(shape, distributedActivationTensorAttr));

    return VPU::DistributedTensorAttr::get(ctx, actTensorDistrModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           numClusters, alignment, uniformDistributedSegments, perClusterMemoryShapes,
                                           perClusterMemoryOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                                           nullptr);
}

vpux::VPU::DistributedTensorAttr vpux::VPU::getConcatExplicitDistributedAttrForNewShape(
        VPU::DistributedTensorAttr originDistribution, vpux::ShapeRef newShape, mlir::MLIRContext* ctx) {
    // For non-overlapped mode, use already existing methods that compute per cluster shapes/methods
    if (originDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return VPU::getConcatExplicitDistributedAttr(
                newShape, originDistribution.getMode().getValue(), originDistribution.getNumTiles(),
                originDistribution.getNumClusters(), originDistribution.getAlignment(), nullptr, nullptr, nullptr,
                originDistribution.getUniformDistributedSegments(), ctx);
    }

    const auto numTiles = vpux::parseIntArrayAttr<int64_t>(originDistribution.getNumTiles());
    auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(originDistribution.getMemoryShapes());

    // For overlapped mode, on the clustering dim, the shapes are taken from the initial distribution, while the rest of
    // the dims will take values from the new shape; this works as long as the concat axis != clustering axis, which is
    // a prerequisite of Distributed Concat
    for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
        for (size_t dim = 0; dim < numTiles.size(); dim++) {
            if (numTiles[dim] == 1) {
                memoryShapes[cluster][dim] = newShape[Dim(dim)];
            }
        }
    }

    auto memoryShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes);
    return VPU::DistributedTensorAttr::get(
            ctx, originDistribution.getMode(), originDistribution.getNumTiles(), originDistribution.getKernel(),
            originDistribution.getPads(), originDistribution.getStrides(), originDistribution.getNumClusters(),
            originDistribution.getAlignment(), originDistribution.getUniformDistributedSegments(), memoryShapesAttr,
            originDistribution.getMemoryOffsets(), memoryShapesAttr, originDistribution.getMemoryOffsets(),
            originDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSliceLikeOps(VPU::DistributedTensorAttr originDistribution,
                                                                          ArrayRef<int64_t> sliceShape,
                                                                          ArrayRef<int64_t> originShape,
                                                                          mlir::MLIRContext* ctx) {
    const auto mode = originDistribution.getMode().getValue();

    // Explicit DistributedAttr can be inferred for Slice in SEGMENTED case or in any case that has full tensor
    // in all cluster (i.e. if mode contains DUPLICATED or SEGMENTED).
    VPUX_THROW_WHEN(
            (mode != VPU::DistributionMode::SEGMENTED) && (mode != VPU::DistributionMode::OVERLAPPED) &&
                    !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
                    !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED),
            "Cannot apply Slice-like Op on input with explicit memory/compute shapes/offsets with DistributionMode {0}",
            originDistribution.getMode());

    // For Overlapped, if slice axis is clustering axis, per cluster shapes/offsets need to be computed taking into
    // consideration Slice/Subview's neighbour ops, which cannot be done with information available here; the calling
    // pass should fill the correct information in this scenario
    VPUX_THROW_WHEN(
            mode == VPU::DistributionMode::OVERLAPPED &&
                    VPU::isSegmentedOverlappedAxisSameAsSliceAxis(originDistribution.getNumTiles(), originShape,
                                                                  sliceShape),
            "Overlapped clustering axis is the same as Slice/Subview axis; cannot infer per cluster shapes/offsets "
            "without compute op information");

    const auto getDistribution = [&](mlir::ArrayAttr perClusterShapesAttr,
                                     mlir::ArrayAttr perClusterOffsetsAttr) -> VPU::DistributedTensorAttr {
        // Slice/SubviewOp is not a "compute" op, so compute shapes/offsets have no reason to be different
        // from memory shapes/offsets
        return VPU::DistributedTensorAttr::get(
                ctx, originDistribution.getMode(), originDistribution.getNumTiles(), originDistribution.getKernel(),
                originDistribution.getPads(), originDistribution.getStrides(), originDistribution.getNumClusters(),
                originDistribution.getAlignment(), originDistribution.getUniformDistributedSegments(),
                perClusterShapesAttr, perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr,
                originDistribution.getEqualMemoryAndComputeView());
    };

    if (mode == VPU::DistributionMode::OVERLAPPED) {
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(originDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < originShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != originShape[dim]) {
                    memoryShapes[cluster][dim] = sliceShape[dim];
                }
            }
        }

        return getDistribution(vpux::getIntArrayOfArray(ctx, memoryShapes), originDistribution.getMemoryOffsets());
    }

    const auto memoryShapes = VPU::getPerClusterMemoryShapes(Shape(sliceShape), originDistribution);
    VPUX_THROW_WHEN(
            !memoryShapes.has_value(),
            "Cannot compute memory shapes for the shape of Slice/Subview's output; shape = {0}, distribution ={1}",
            sliceShape, originDistribution);

    auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes.value());
    auto perClusterOffsetsAttr =
            vpux::getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(Shape(sliceShape), originDistribution));
    auto distribution = getDistribution(perClusterShapesAttr, perClusterOffsetsAttr);

    if (originDistribution.getAlignment() == nullptr) {
        return distribution;
    }

    const auto isAlignmentDim = [](auto dim) {
        return dim > 1;
    };

    const auto alignment = parseIntArrayAttr<int64_t>(originDistribution.getAlignment());
    auto alignIt = llvm::find_if(alignment, isAlignmentDim);
    if (alignIt == alignment.end()) {
        return distribution;
    }

    const auto axis = std::distance(alignment.begin(), alignIt);
    if (sliceShape[axis] == originShape[axis]) {
        return distribution;
    }

    // If there is alignment on the Slice axis, discard it, as it's not ensured that the new Slice output shape can be
    // aligned. This could only happen for a full memory mode (DUPLICATED/MULTICASTED) as SEGMENTED is forbidden on the
    // same axis as the Slice.
    auto unalignedDistribution = VPU::DistributedTensorAttr::get(
            ctx, originDistribution.getMode(), originDistribution.getNumTiles(), originDistribution.getKernel(),
            originDistribution.getPads(), originDistribution.getStrides(), originDistribution.getNumClusters(), nullptr,
            originDistribution.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr,
            originDistribution.getEqualMemoryAndComputeView());

    const auto unAlignedMemoryShapes = VPU::getPerClusterMemoryShapes(Shape(sliceShape), unalignedDistribution);
    VPUX_THROW_WHEN(
            !unAlignedMemoryShapes.has_value(),
            "Cannot compute memory shapes for the shape of Slice/Subview's output; shape = {0}, distribution ={1}",
            sliceShape, unalignedDistribution);

    auto unalignedClusterShapesAttr = vpux::getIntArrayOfArray(ctx, unAlignedMemoryShapes.value());
    auto unalignedClusterOffsetsAttr = vpux::getIntArrayOfArray(
            ctx, VPU::getPerClusterMemoryShapeOffsets(Shape(sliceShape), unalignedDistribution));

    return VPU::DistributedTensorAttr::get(
            ctx, originDistribution.getMode(), originDistribution.getNumTiles(), originDistribution.getKernel(),
            originDistribution.getPads(), originDistribution.getStrides(), originDistribution.getNumClusters(), nullptr,
            originDistribution.getUniformDistributedSegments(), unalignedClusterShapesAttr, unalignedClusterOffsetsAttr,
            unalignedClusterShapesAttr, unalignedClusterOffsetsAttr, originDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getNonOverlappedDistributedAttr(
        ShapeRef shape, VPU::DistributionModeAttr distrModeAttr, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        mlir::MLIRContext* ctx) {
    VPUX_THROW_WHEN(distrModeAttr.getValue() == VPU::DistributionMode::OVERLAPPED,
                    "getNonOverlappedDistributedAttr: distribution mode is OVERLAPPED");
    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters, alignment, uniformDistributedSegments,
            nullptr, nullptr, nullptr, nullptr, nullptr);
    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensorAttr);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensorAttr);
    auto perClusterMemoryShapes = getIntArrayOfArray(ctx, optionalClusterMemoryShapes.value());
    auto perClusterMemoryOffsets =
            getIntArrayOfArray(ctx, VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensorAttr));
    auto perClusterComputeShapes =
            getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapes(shape, distributedTensorAttr));
    auto perClusterComputeOffsets =
            getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapeOffsets(shape, distributedTensorAttr));

    return VPU::DistributedTensorAttr::get(ctx, distrModeAttr, numTiles, nullptr, nullptr, nullptr, numClusters,
                                           alignment, uniformDistributedSegments, perClusterComputeShapes,
                                           perClusterComputeOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                                           nullptr);
}

NDTypeInterface vpux::VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(NDTypeInterface buff, ShapeRef shape,
                                                                              mlir::Type elemType) {
    auto distributedBuff = mlir::dyn_cast<VPUIP::DistributedBufferType>(buff);
    VPUX_THROW_WHEN(distributedBuff == nullptr,
                    "changeShapeElemTypeForNonOverlappedDistributedBuffers: buff is not DistributedBufferType = {0}",
                    buff);

    auto distribution = distributedBuff.getDistribution();
    VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "DistributedBuffer has mode different from DUPLICATED after unrolling");
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedBuff.getDistribution())) {
        auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                shape, distribution.getMode(), nullptr, distribution.getNumClusters(), nullptr,
                distribution.getUniformDistributedSegments(), distributedBuff.getContext());
        return distributedBuff.changeShapeElemTypeForExplicitDistribution(shape, elemType, newDistribution);
    }

    return distributedBuff.changeShapeElemType(shape, elemType);
};

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSparseData(
        VPU::DistributedTensorAttr denseDataDistribution, ShapeRef dataShape, VPU::SEAttr seAttr,
        mlir::MLIRContext* ctx) {
    if (seAttr == nullptr) {
        return denseDataDistribution;
    }

    SmallVector<int64_t> seAttrOffsets(dataShape.size(), 0);
    if (auto tileInfo = seAttr.getTileInfo(); tileInfo.has_value() && tileInfo->offsets != nullptr) {
        seAttrOffsets = parseIntArrayAttr<int64_t>(tileInfo->offsets);
    }

    auto getDataShapesOffsets =
            [&](mlir::ArrayAttr denseDataShapesAttr,
                mlir::ArrayAttr denseDataOffsetsAttr) -> std::pair<mlir::ArrayAttr, mlir::ArrayAttr> {
        const auto denseDataShapes = parseIntArrayOfArrayAttr<int64_t>(denseDataShapesAttr);
        const auto denseDataOffsets = parseIntArrayOfArrayAttr<int64_t>(denseDataOffsetsAttr);
        const auto clusterNum = denseDataShapes.size();
        auto dataShapesVec = SmallVector<SmallVector<int64_t>>(clusterNum);
        auto dataOffsetsVec = SmallVector<SmallVector<int64_t>>(clusterNum);

        for (size_t clusterIdx = 0; clusterIdx < clusterNum; ++clusterIdx) {
            const auto denseDataShape = Shape(denseDataShapes[clusterIdx]);
            const auto denseDataOffset = Shape(denseDataOffsets[clusterIdx]);

            SmallVector<int64_t> startOffsets(denseDataShape.size());
            std::transform(denseDataOffset.begin(), denseDataOffset.end(), seAttrOffsets.begin(), startOffsets.begin(),
                           [](const int64_t dataOffset, const int64_t seOffset) {
                               return dataOffset + seOffset;
                           });

            const auto dataStartOffsets = seAttr.backInferInputCoord(Shape(startOffsets), dataShape).raw();
            dataOffsetsVec[clusterIdx] = dataStartOffsets;

            SmallVector<int64_t> endOffsets(denseDataShape.size());
            std::transform(startOffsets.begin(), startOffsets.end(), denseDataShape.begin(), endOffsets.begin(),
                           [](const int64_t start, const int64_t size) {
                               return start + size - 1;
                           });

            const auto dataEndOffsets = seAttr.backInferInputCoord(Shape(endOffsets), dataShape).raw();

            SmallVector<int64_t> dataShapes(denseDataShape.size());
            std::transform(dataStartOffsets.begin(), dataStartOffsets.end(), dataEndOffsets.begin(), dataShapes.begin(),
                           [](const int64_t start, const int64_t end) {
                               return end - start + 1;
                           });
            dataShapesVec[clusterIdx] = dataShapes;
        }

        return {getIntArrayOfArray(ctx, dataShapesVec), getIntArrayOfArray(ctx, dataOffsetsVec)};
    };

    const auto computeView =
            getDataShapesOffsets(denseDataDistribution.getComputeShapes(), denseDataDistribution.getComputeOffsets());
    const auto memoryView =
            getDataShapesOffsets(denseDataDistribution.getMemoryShapes(), denseDataDistribution.getMemoryOffsets());

    return VPU::DistributedTensorAttr::get(ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(),
                                           nullptr, nullptr, nullptr, denseDataDistribution.getNumClusters(),
                                           /*alignment*/ nullptr, denseDataDistribution.getUniformDistributedSegments(),
                                           computeView.first, computeView.second, memoryView.first, memoryView.second,
                                           denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSparsityMap(
        VPU::DistributedTensorAttr denseDataDistribution, ShapeRef sparsityMapShape, mlir::UnitAttr isWeights,
        mlir::MLIRContext* ctx) {
    if (isWeights == nullptr) {
        return denseDataDistribution;
    }

    auto isValidDistributionForWeights = [&]() -> bool {
        if (denseDataDistribution.getNumTiles() == nullptr) {
            return true;
        }

        const auto numTiles = parseIntArrayAttr<int64_t>(denseDataDistribution.getNumTiles());
        if (numTiles.size() == 4 && numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] == 1 &&
            numTiles[Dims4D::Act::W.ind()] == 1) {
            return true;
        }

        return false;
    };

    VPUX_THROW_WHEN(!isValidDistributionForWeights(),
                    "Weights should be segmented only over OC dim, distributed attr = {0}", denseDataDistribution);

    auto getWeightsShapes = [&](mlir::ArrayAttr shapesAttr) -> mlir::ArrayAttr {
        auto shapesVec = parseIntArrayOfArrayAttr<int64_t>(shapesAttr);

        for (auto& shapes : shapesVec) {
            shapes[Dims4D::Filter::IC.ind()] = sparsityMapShape[Dims4D::Filter::IC];
            shapes[Dims4D::Filter::KY.ind()] = sparsityMapShape[Dims4D::Filter::KY];
            shapes[Dims4D::Filter::KX.ind()] = sparsityMapShape[Dims4D::Filter::KX];
        }

        return getIntArrayOfArray(ctx, shapesVec);
    };

    return VPU::DistributedTensorAttr::get(
            ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(), nullptr, nullptr, nullptr,
            denseDataDistribution.getNumClusters(), denseDataDistribution.getAlignment(),
            denseDataDistribution.getUniformDistributedSegments(),
            getWeightsShapes(denseDataDistribution.getComputeShapes()), denseDataDistribution.getComputeOffsets(),
            getWeightsShapes(denseDataDistribution.getMemoryShapes()), denseDataDistribution.getMemoryOffsets(),
            denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributedTensorAttr vpux::VPU::getExplicitDistrAttrForSETable(VPU::DistributedTensorAttr denseDataDistribution,
                                                                     const size_t seSize, mlir::MLIRContext* ctx) {
    auto getSETableShapesOffsets = [&](mlir::ArrayAttr shapesOffsetsAttr,
                                       const bool isOffset = false) -> mlir::ArrayAttr {
        auto shapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(shapesOffsetsAttr);

        for (auto& shapesOffsets : shapesOffsetsVec) {
            // In cases where tensor is SEGMENTED over C, SETable depth per cluster must be adjusted
            shapesOffsets[Dims4D::Act::C.ind()] =
                    isOffset ? shapesOffsets[Dims4D::Act::C.ind()] / static_cast<int64_t>(seSize)
                             : divUp(shapesOffsets[Dims4D::Act::C.ind()], static_cast<int64_t>(seSize));
        }

        return getIntArrayOfArray(ctx, shapesOffsetsVec);
    };

    return VPU::DistributedTensorAttr::get(ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(),
                                           nullptr, nullptr, nullptr, denseDataDistribution.getNumClusters(),
                                           denseDataDistribution.getAlignment(),
                                           denseDataDistribution.getUniformDistributedSegments(),
                                           getSETableShapesOffsets(denseDataDistribution.getComputeShapes()),
                                           getSETableShapesOffsets(denseDataDistribution.getComputeOffsets(), true),
                                           getSETableShapesOffsets(denseDataDistribution.getMemoryShapes()),
                                           getSETableShapesOffsets(denseDataDistribution.getMemoryOffsets(), true),
                                           denseDataDistribution.getEqualMemoryAndComputeView());
}
