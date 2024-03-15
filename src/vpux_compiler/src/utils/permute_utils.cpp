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

bool vpux::isTrivialReorder(DimsOrder inOrder, DimsOrder outOrder, ShapeRef shape) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    const auto shapeIsOne = [&](const Dim& perm) -> bool {
        return shape[perm] == 1;
    };
    // Ignore dim whose shape is one
    inPerm.erase(std::remove_if(inPerm.begin(), inPerm.end(), shapeIsOne), inPerm.end());
    outPerm.erase(std::remove_if(outPerm.begin(), outPerm.end(), shapeIsOne), outPerm.end());

    return inPerm == outPerm;
}

bool vpux::isTrivialReorder(IE::ReorderOp origOp) {
    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    const auto inShape = getShape(origOp.getInput());

    return isTrivialReorder(inOrder, outOrder, inShape);
}

mlir::AffineMap vpux::getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }

    return mlir::AffineMap::getPermutationMap(ArrayRef(memPerm), ctx);
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

// change order like from CNHW to NCHW
DimsOrder vpux::moveD0ToTheFront(DimsOrder inOrder) {
    SmallVector<vpux::Dim> perm = {Dim(0)};
    auto permutation = inOrder.toPermutation();
    std::copy_if(permutation.begin(), permutation.end(), std::back_inserter(perm), [](const Dim dim) {
        return dim != Dim(0);
    });
    return DimsOrder::fromPermutation(ArrayRef(perm));
}

VPU::DistributedTensorAttr vpux::applyPermutationOnDistributedTensorAttr(VPU::DistributedTensorAttr inDistribution,
                                                                         mlir::AffineMap memPerm, DimsOrder srcOrder,
                                                                         DimsOrder dstOrder, ShapeRef srcShape,
                                                                         ShapeRef dstShape) {
    auto ctx = inDistribution.getContext();

    auto permuteAxisOfArray = [&](ArrayRef<int64_t> arr) -> SmallVector<int64_t> {
        // At VPUIP level, VPU.LayoutCast gets lowered to VPUIP.PermuteCast.
        // LayoutCast will have same in/out shape but different orders, which cannot be handled
        // the same way as the VPU.PermuteCast ops which have the same memory shape between input
        // and output even if orders and logical shapes differ. In such a case, applying the
        // `toMemoryOrder -> applyPerm -> toLogicalOrder` transformations will not permute the
        // distributed attr correctly.
        if (srcShape == dstShape) {
            return SmallVector<int64_t>(arr);
        }

        const auto arrInMemOrder = srcOrder.toMemoryOrder(Shape(arr));
        const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, memPerm);
        const auto arrPermutedInLogicalOrder = dstOrder.toLogicalOrder(arrPermutedInMemOrder).raw();

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

// Normalize permutation vector
// Example: [1, 3, 7, 6] -> [0, 1, 3, 2]
void normalizePermutation(SmallVector<uint32_t>& vec) {
    SmallVector<uint32_t> sorted(vec);
    llvm::DenseMap<uint32_t, uint32_t> helper;
    std::sort(sorted.begin(), sorted.end());

    for (size_t i = 0; i < sorted.size(); ++i) {
        helper.insert(std::make_pair(sorted[i], checked_cast<uint32_t>(i)));
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = helper[vec[i]];
    }
}

std::pair<SmallVector<uint32_t>, SmallVector<int64_t>> vpux::getMergedPermutationAndShape(NDTypeInterface inputType,
                                                                                          mlir::AffineMap permutation,
                                                                                          int64_t rank) {
    auto memShape = to_small_vector(inputType.getMemShape());
    auto origPermVec = DimsOrder::fromAffineMap(permutation).toPermutation();

    // Example of origPermVec
    // origPermVec = [d0, d1, d2, d3] -> [d1, d2, d3, d0];
    // origPermVec[0] = 1, origPermVec[1] = 2, origPermVec[2] = 3, origPermVec[3] = 0
    SmallVector<uint32_t> permVec;
    SmallVector<int64_t> shapeVec;

    for (auto d : origPermVec) {
        // Dims with size 1 are dropped
        if (memShape[d.ind()] != 1) {
            permVec.push_back(checked_cast<uint32_t>(d.ind()));
        }
    }

    for (auto dimSize : memShape) {
        // Dims with size 1 are dropped
        if (dimSize != 1) {
            shapeVec.push_back(dimSize);
        }
    }

    normalizePermutation(permVec);

    if ((int64_t)shapeVec.size() < rank) {
        return std::make_pair(permVec, shapeVec);
    }

    // Merge dims that are adjacent before and after permutation
    // Example:
    // memShape =[2, 4, 25, 255, 255]
    // permVec = [d0, d1, d2, d3, d4] -> [d0, d4, d1, d2, d3]
    //
    // mergedShape = [2, 25500, 255]
    // mergedPermutation = [d0, d1, d2] -> [d0, d2, d1]
    SmallVector<uint32_t> mergedPermutation;
    SmallVector<int64_t> mergedShape;

    std::map<uint32_t, int64_t> mergedShapeMap;
    size_t j = 0;
    int64_t remainingRank = permVec.size();
    for (size_t i = 0; i < checked_cast<size_t>(permVec.size()); i = j) {
        int64_t dimSize = shapeVec[permVec[i]];
        for (j = i + 1; j < checked_cast<size_t>(permVec.size()) && (permVec[j - 1] + 1 == permVec[j]); ++j) {
            if (remainingRank < rank) {
                break;
            }
            dimSize *= shapeVec[permVec[j]];
            remainingRank--;
        }

        mergedShapeMap.insert(std::make_pair(permVec[i], dimSize));
        mergedPermutation.push_back(permVec[i]);
    }

    // Keys iterated in ascending order
    for (const auto& p : mergedShapeMap) {
        mergedShape.push_back(p.second);
    }

    // Normalize vectors
    normalizePermutation(mergedPermutation);

    return std::make_pair(mergedPermutation, mergedShape);
}

void vpux::extendPermutationAndShape(SmallVector<uint32_t>& permutation, SmallVector<int64_t>& shape, int64_t rank) {
    // Padding to rank dimension if needed
    // Here always padding to the front of the shape/perm vector. Dim N will be 1, so this mempermute will be
    // converted to MaxPool in ConvertMemPermuteToPoolPass. Otherwise if N > 1, the mempermute will always be
    // PermuteDMA.
    // For example:
    // shape: [4, 128, 256], permutation: [d1, d0, d2]
    // will pad like below:
    // shape: [1, 4, 128, 256], permutation: [d0, d2, d1, d3]
    // If padding to the end, shape will be [4, 128, 256, 1] with permutation [d1, d0, d2, d3]
    int64_t padSize = rank - checked_cast<int64_t>(permutation.size());
    if (padSize > 0) {
        auto paddedPermutation = SmallVector<uint32_t>(rank);
        auto paddedShape = SmallVector<int64_t>(rank);
        for (int64_t i = 0; i < rank; ++i) {
            paddedPermutation[i] = i < padSize ? i : permutation[i - padSize] + padSize;
            paddedShape[i] = i < padSize ? 1 : shape[i - padSize];
        }
        permutation.assign(paddedPermutation);
        shape.assign(paddedShape);
    }
}

IE::LayerWithPermuteInterface vpux::getFusableLayerWithPermuteInterface(mlir::Operation* op) {
    auto inputOp = op->getOperand(0).getDefiningOp();
    if (auto quantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(inputOp)) {
        auto outElemType = quantizeCastOp.getOutput().getType().getElementType();
        if (quantizeCastOp->hasOneUse() && outElemType.isa<mlir::quant::UniformQuantizedType>()) {
            inputOp = quantizeCastOp.getInput().getDefiningOp();
        }
    }
    return mlir::dyn_cast_or_null<IE::LayerWithPermuteInterface>(inputOp);
}
