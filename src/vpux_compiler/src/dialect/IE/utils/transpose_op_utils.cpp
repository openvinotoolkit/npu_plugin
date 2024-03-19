//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"

namespace vpux {
namespace IE {

// Consider the following tensor transposition:
// 1x16x32x64 -> 1x64x16x32:
// 1    16  32  64  ->  1   64  16  32
// d0   d1  d2  d3  ->  d0  d3  d1  d2
// This transposition is described as:
// (d0, d1, d2, d3) -> (d0, d3, d1, d2)
// To find out the order of target tensor, one must inverse applied affine map:
// d0, d3, d1, d2   ->  d0, d1, d2, d3
// aN, aC, aH, aW   ->  aN, aH, aW, aC
// Thus, target order of dimensions is actually NHWC.
DimsOrder deduceInverseOrder(IE::TransposeOp op) {
    const auto orderAttr = op.getOrderValueAttr();
    const auto order = DimsOrder::fromAffineMap(orderAttr.getValue());

    // Given the example above, inputOrder is NCHW.
    const auto inputOrder = vpux::DimsOrder::fromValue(op.getInput());
    // The order is (d0, d1, d2, d3) -> (d0, d3, d1, d2), NWCH
    auto targetPermutation = order.toPermutation();
    // The following loop in case of NCHW just goes over d0(N), d1(C), d2(H), d3(W).
    // N is trivial enough (it is d0 in both NCHW and NWCH).
    // Now C resides at index 1 in NCHW and index 2 in NWCH, so targetPermutation[1] = d2
    // Applying this for H gives: index 2 in NCHW and index 3 in NWCH => targetPermutation[2] = d3
    // Finally, for W: index 3 in NCHW and index 1 in NWCH => targetPermutation[2] = d1
    // The result is: d0, d2, d3, d1
    for (const auto& perm : inputOrder.toPermutation()) {
        const auto permInd = perm.ind();
        targetPermutation[permInd] = Dim(order.dimPos(Dim(permInd)));
    }

    return DimsOrder::fromPermutation(targetPermutation);
}

bool isWHSwappingTranspose(IE::TransposeOp op) {
    const auto orderAttr = op.getOrderValueAttr();
    const auto order = DimsOrder::fromAffineMap(orderAttr.getValue());
    const auto inputOrder = vpux::DimsOrder::fromValue(op.getInput());
    size_t dimW, dimH;

    if (inputOrder.numDims() == 4) {
        dimW = inputOrder.dimPos(Dim(3));
        dimH = inputOrder.dimPos(Dim(2));
    } else if (inputOrder.numDims() == 3) {
        dimW = inputOrder.dimPos(Dim(2));
        dimH = inputOrder.dimPos(Dim(1));
    } else {
        return false;
    }

    if (order.dimAt(dimW) == inputOrder.dimAt(dimH) && order.dimAt(dimH) == inputOrder.dimAt(dimW)) {
        return true;
    }

    return false;
}

}  // namespace IE
}  // namespace vpux
