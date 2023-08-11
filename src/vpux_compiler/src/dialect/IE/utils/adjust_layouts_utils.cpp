//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <vpux/compiler/utils/adjust_layout_utils.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

using namespace vpux;

namespace vpux {

void insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                           mlir::PatternRewriter& rewriter, Logger log) {
    auto curOrder = DimsOrder::fromValue(input.get());
    log.trace("Insert ReorderOp: curOrder = {0}, dstOrder = {1}", curOrder, dstOrder);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto reorderOp =
            rewriter.create<IE::ReorderOp>(op->getLoc(), input.get(), dstOrder.toAffineMap(rewriter.getContext()));

    log.trace("Redirect input to the new Value");
    input.set(reorderOp.output());
}

IE::ReorderOp insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                     mlir::PatternRewriter& rewriter, Logger log) {
    auto curOrder = DimsOrder::fromValue(output);
    log.trace("Insert ReorderOp: curOrder = {0}, dstOrder = {1}", curOrder, dstOrder);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    auto reorderOp = rewriter.create<IE::ReorderOp>(op->getLoc(), output, dstOrder.toAffineMap(rewriter.getContext()));

    log.trace("Redirect output users to the new Value");
    output.replaceAllUsesExcept(reorderOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{reorderOp});

    return reorderOp;
}

void changeDimsOrder(mlir::Value val, DimsOrder newOrder, Logger log) {
    const auto origType = val.getType().cast<vpux::NDTypeInterface>();
    const auto newType = origType.changeDimsOrder(newOrder);

    log.trace("Change Value type to '{0}'", newType);
    val.setType(newType);
}

}  // namespace vpux
