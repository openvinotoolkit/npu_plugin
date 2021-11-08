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
    MemShape outShape(memShape.size());

    const auto perm = DimsOrder::fromAffineMap(memPerm);
    auto indices = to_small_vector(irange(perm.numDims()) | transformed([&](int64_t idx) {
                                       return checked_cast<int64_t>(perm.dimAt(idx).ind());
                                   }));

    for (size_t idx = 0; idx < memShape.size(); ++idx) {
        const auto d_in = MemDim(indices[idx]);
        const auto d_out = MemDim(idx);
        outShape[d_out] = memShape[d_in];
    }
    return outShape;
}

bool vpux::isTrivial(const ShapeRef shape) {
    const auto nonTrivialPredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    return std::count_if(shape.begin(), shape.end(), nonTrivialPredicate) == 1;
}

bool vpux::isShapeNotTrivAndIsPermNotIdentity(const ShapeRef shape, mlir::AffineMap memPerm) {
    return (!isTrivial(shape) && !memPerm.isIdentity());
}
