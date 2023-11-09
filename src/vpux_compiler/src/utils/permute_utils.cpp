//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

MemShape vpux::applyPerm(MemShapeRef memShape, mlir::AffineMap memPerm) {
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(memShape.size() == perm.numDims(), "Permutation '{0}' is not compatible with shape '{1}'",
                      memPerm, memShape);

    MemShape outShape(memShape.size());

    for (auto ind : irange(outShape.size())) {
        const auto outDim = MemDim(ind);
        const auto inDim = MemDim(perm.dimAt(ind).ind());
        outShape[outDim] = memShape[inDim];
    }

    return outShape;
}

bool vpux::isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm) {
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(inShape.size() == perm.numDims(), "Permutation '{0}' is not compatible with shape '{1}'", memPerm,
                      inShape);

    SmallVector<int64_t> nonTrivialPerm;

    for (auto ind : irange(inShape.size())) {
        const auto inDim = MemDim(perm.dimAt(ind).ind());

        if (inShape[inDim] == 1) {
            continue;
        }

        nonTrivialPerm.push_back(inDim.ind());
    }

    if (nonTrivialPerm.empty()) {
        return true;
    }

    for (auto ind : irange<size_t>(1, nonTrivialPerm.size())) {
        if (nonTrivialPerm[ind] < nonTrivialPerm[ind - 1]) {
            return false;
        }
    }

    return true;
}

SmallVector<int64_t> vpux::getPermutateDims(MemShapeRef inShape, mlir::AffineMap memPerm) {
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(inShape.size() == perm.numDims(), "Permutation '{0}' is not compatible with shape '{1}'", memPerm,
                      inShape);

    SmallVector<int64_t> permutateDims;

    for (auto ind : irange(inShape.size())) {
        const auto inDim = MemDim(perm.dimAt(ind).ind());

        if (inShape[inDim] == 1) {
            continue;
        }

        permutateDims.push_back(inDim.ind());
    }

    for (auto ind : irange<size_t>(1, permutateDims.size())) {
        if (permutateDims[ind] > permutateDims[ind - 1]) {
            permutateDims.clear();
            return permutateDims;
        }
    }

    return permutateDims;
}

bool vpux::isTrivialReorder(IE::ReorderOp origOp) {
    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto outOrder = DimsOrder::fromValue(origOp.output());
    const auto inShape = getShape(origOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    const auto memPerm = getPermutationFromOrders(inOrder, outOrder, origOp->getContext());
    if (isTrivialPermute(inMemShape, memPerm)) {
        return true;
    }

    return false;
}

mlir::AffineMap vpux::getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }

    return mlir::AffineMap::getPermutationMap(makeArrayRef(memPerm), ctx);
}

DimsOrder vpux::applyPermutation(const DimsOrder srcOrder, const DimsOrder dstOrder) {
    const auto srcPermutation = srcOrder.toPermutation();
    const auto dstPermutation = dstOrder.toPermutation();
    DimArr result;
    const auto getDimAt = [&](const Dim& perm) -> Dim {
        return srcOrder.dimAt(perm.ind());
    };
    std::transform(dstPermutation.begin(), dstPermutation.end(), std::back_inserter(result), getDimAt);
    return DimsOrder::fromPermutation(result);
}

VPU::DistributedTensorAttr vpux::applyPermutationOnDistributedTensorAttr(VPU::DistributedTensorAttr inDistribution,
                                                                         mlir::AffineMap memPerm, DimsOrder inOrder,
                                                                         DimsOrder outOrder) {
    auto ctx = inDistribution.getContext();

    auto permuteAxisOfArray = [&](ArrayRef<int64_t> arr) -> SmallVector<int64_t> {
        const auto arrInMemOrder = inOrder.toMemoryOrder(Shape(arr));
        const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, memPerm);
        const auto arrPermutedInLogicalOrder = outOrder.toLogicalOrder(arrPermutedInMemOrder).raw();

        return arrPermutedInLogicalOrder;
    };

    auto numTilesAttr = inDistribution.getNumTiles();
    if (numTilesAttr != nullptr) {
        const auto numTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
        numTilesAttr = getIntArrayAttr(ctx, permuteAxisOfArray(numTilesVec));
    }

    auto alignmentAttr = inDistribution.getAlignment();
    if (alignmentAttr != nullptr) {
        const auto alignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
        alignmentAttr = getIntArrayAttr(ctx, permuteAxisOfArray(alignmentVec));
    }

    auto permutePerClusterShapesOffsets = [&](mlir::ArrayAttr shapesOffsetsAttr) -> mlir::ArrayAttr {
        const auto inPerClusterShapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(shapesOffsetsAttr);
        auto outComputeShapesVec = SmallVector<SmallVector<int64_t>>();

        for (const auto& shapesOffsets : inPerClusterShapesOffsetsVec) {
            outComputeShapesVec.push_back(permuteAxisOfArray(shapesOffsets));
        }

        return getIntArrayOfArray(ctx, outComputeShapesVec);
    };

    auto computeShapesAttr = (inDistribution.getComputeShapes() != nullptr)
                                     ? permutePerClusterShapesOffsets(inDistribution.getComputeShapes())
                                     : inDistribution.getComputeShapes();

    auto computeOffsetsAttr = (inDistribution.getComputeOffsets() != nullptr)
                                      ? permutePerClusterShapesOffsets(inDistribution.getComputeOffsets())
                                      : inDistribution.getComputeOffsets();

    auto memoryShapesAttr = (inDistribution.getMemoryShapes() != nullptr)
                                    ? permutePerClusterShapesOffsets(inDistribution.getMemoryShapes())
                                    : inDistribution.getMemoryShapes();

    auto memoryOffsetsAttr = (inDistribution.getMemoryOffsets() != nullptr)
                                     ? permutePerClusterShapesOffsets(inDistribution.getMemoryOffsets())
                                     : inDistribution.getMemoryOffsets();

    return VPU::DistributedTensorAttr::get(
            ctx, inDistribution.getMode(), numTilesAttr, inDistribution.getKernel(), inDistribution.getPads(),
            inDistribution.getStrides(), inDistribution.getNumClusters(), alignmentAttr,
            inDistribution.getUniformDistributedSegments(), computeShapesAttr, computeOffsetsAttr, memoryShapesAttr,
            memoryOffsetsAttr, inDistribution.getEqualMemoryAndComputeView());
}
