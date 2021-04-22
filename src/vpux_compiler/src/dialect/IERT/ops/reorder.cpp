//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
