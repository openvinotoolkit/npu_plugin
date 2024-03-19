//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/softmax_utils.hpp"

namespace vpux {
namespace IE {

// Check the axis of SoftMaxOp has been in the last memory dimension which is most efficient
bool isSoftMaxAxisInLastMemDim(IE::SoftMaxOp op) {
    const auto inputRank = op.getInput().getType().cast<vpux::NDTypeInterface>().getRank();
    const auto inputOrder = DimsOrder::fromValue(op.getInput());
    const auto axisMemPos = inputOrder.toMemDim(Dim(op.getAxisInd()));
    if (axisMemPos.ind() == inputRank - 1) {
        return true;
    }

    return false;
}

}  // namespace IE
}  // namespace vpux
