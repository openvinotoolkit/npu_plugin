//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
using namespace vpux;

bool IE::isLegalReorderAddPattern(IE::ReorderOp origOp) {
    if (origOp.getOutput().use_empty()) {
        return false;
    }

    auto opNce = *origOp.getOutput().getUsers().begin();
    // check just 1 child for linked patern
    for (auto user : llvm::make_early_inc_range(origOp.getResult().getUsers())) {
        if (user != opNce) {
            return false;
        }
    }

    if (auto opAdd = mlir::dyn_cast<IE::AddOp>(opNce)) {
        if (opAdd.getInput1() != opAdd.getInput2()) {
            return false;
        }
        if (!opAdd.getOutput().hasOneUse()) {
            return false;
        }
        if (!mlir::isa<IE::QuantizeCastOp>(*opAdd.getOutput().getUsers().begin())) {
            return false;
        }
        return true;
    }

    return false;
}

bool IE::isLegalReorderAvgPoolPattern(IE::ReorderOp origOp) {
    if (!origOp.getOutput().hasOneUse()) {
        return false;
    }
    auto opNce = *origOp.getOutput().getUsers().begin();
    if (auto opPooling = mlir::dyn_cast<IE::AvgPoolOp>(opNce)) {
        return vpux::IE::isQuantizedPurposeAvgPool(opPooling);
    }

    return false;
}
