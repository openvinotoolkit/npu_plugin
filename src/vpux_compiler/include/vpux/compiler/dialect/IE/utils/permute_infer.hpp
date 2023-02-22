//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool strictInfer);

template <typename PermOpPrev, typename PermOp>
mlir::LogicalResult fusePermutations(PermOp permuteOp, mlir::PatternRewriter& rewriter) {
    auto prevPermuteOp = mlir::dyn_cast_or_null<PermOpPrev>(permuteOp.input().getDefiningOp());
    if (prevPermuteOp == nullptr) {
        return mlir::failure();
    }

    auto prevMemPerm = prevPermuteOp.mem_perm();
    auto memPerm = permuteOp.mem_perm();
    auto newMemPerm = memPerm.compose(prevMemPerm);

    const auto canFuseIntoPermuteCastOp =
            mlir::isa<IE::PermuteCastOp>(prevPermuteOp) && mlir::isa<IE::PermuteCastOp>(permuteOp);
    if (canFuseIntoPermuteCastOp) {
        rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(permuteOp, permuteOp.getType(), prevPermuteOp.input(),
                                                       permuteOp.dst_orderAttr(), mlir::AffineMapAttr::get(newMemPerm));
    } else {
        rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(permuteOp, permuteOp.getType(), prevPermuteOp.input(),
                                                      permuteOp.dst_orderAttr(), mlir::AffineMapAttr::get(newMemPerm));
    }

    return mlir::success();
}
