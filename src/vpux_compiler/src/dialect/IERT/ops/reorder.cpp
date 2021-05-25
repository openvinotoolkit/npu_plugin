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

#include "vpux/compiler/dialect/IERT/ops.hpp"

using namespace vpux;

//
// FoldReorder
//

namespace {

class FoldReorder final : public mlir::OpRewritePattern<IERT::ReorderOp> {
public:
    using mlir::OpRewritePattern<IERT::ReorderOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IERT::ReorderOp reorderOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FoldReorder::matchAndRewrite(IERT::ReorderOp reorderOp, mlir::PatternRewriter& rewriter) const {
    const auto input = reorderOp.input();
    const auto output = reorderOp.output();
    const auto output_buff = reorderOp.output_buff();

    if (input.getType() != output.getType()) {
        return mlir::failure();
    }

    if (output_buff.isa<mlir::BlockArgument>()) {
        // In this case, output of Reorder is an alias of one of the arguments and can be used by Return op. directly.
        // The Return's operand must always be an alias of the result buffers from the argument and we can't just set
        // input in Return operand.
        // So, let's replace the operation with the Copy and a special pass will eliminate it if needed
        rewriter.replaceOpWithNewOp<IERT::CopyOp>(reorderOp, input, output_buff);
        return mlir::success();
    }

    rewriter.replaceOp(reorderOp, {input});

    return mlir::success();
}

}  // namespace

namespace {

#include <vpux/compiler/dialect/IERT/rewriters/generated/reorder.hpp.inc>

}  // namespace

void vpux::IERT::ReorderOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    populateWithGenerated(patterns);
    patterns.insert<FoldReorder>(context);
}
