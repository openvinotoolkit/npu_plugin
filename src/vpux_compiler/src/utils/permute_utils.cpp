//
// Copyright 2021 Intel Corporation.
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
        if (nonTrivialPerm[ind] != nonTrivialPerm[ind - 1] + 1) {
            return false;
        }
    }

    return true;
}
