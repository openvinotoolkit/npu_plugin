//
// Copyright Intel Corporation.
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
