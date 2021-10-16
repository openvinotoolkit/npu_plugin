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

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

void inferPermuteReturnTypeComponents(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                                      SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                      bool useInMemSpace);

template <typename PermOpPrev, typename PermOp>
mlir::LogicalResult fusePermutations(PermOp permuteOp, mlir::PatternRewriter& rewriter) {
    if (!permuteOp.input().hasOneUse()) {
        return mlir::failure();
    }

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
