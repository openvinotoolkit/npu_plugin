//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
