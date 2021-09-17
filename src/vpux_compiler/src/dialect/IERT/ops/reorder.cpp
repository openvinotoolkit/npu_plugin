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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// Canonicalization
//

namespace {

class ReorderToCopy final : public mlir::OpRewritePattern<IERT::ReorderOp> {
public:
    using mlir::OpRewritePattern<IERT::ReorderOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IERT::ReorderOp reorderOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ReorderToCopy::matchAndRewrite(IERT::ReorderOp reorderOp, mlir::PatternRewriter& rewriter) const {
    const auto input = reorderOp.input();
    const auto output = reorderOp.output();
    const auto output_buff = reorderOp.output_buff();

    if (input.getType() != output.getType()) {
        return matchFailed(rewriter, reorderOp, "Reorder is not redundant");
    }
    if (!output_buff.isa<mlir::BlockArgument>()) {
        return matchFailed(rewriter, reorderOp, "Reorder output is not a block argument");
    }

    // In this case, output of Reorder is an alias of one of the arguments and can be used by Return op. directly.
    // The Return's operand must always be an alias of the result buffers from the argument and we can't just set
    // input in Return operand.
    // So, let's replace the operation with the Copy and a special pass will eliminate it if needed
    rewriter.replaceOpWithNewOp<IERT::CopyOp>(reorderOp, input, output_buff);
    return mlir::success();
}

}  // namespace

namespace {

//
// ConvertTrivialReorder
//

class ConvertTrivialReorder final : public mlir::OpRewritePattern<IERT::ReorderOp> {
public:
    using mlir::OpRewritePattern<IERT::ReorderOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IERT::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;
};

bool isTrivial(const ShapeRef shape) {
    const auto nonTrivialPredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    return std::count_if(shape.begin(), shape.end(), nonTrivialPredicate) == 1;
}

mlir::LogicalResult ConvertTrivialReorder::matchAndRewrite(IERT::ReorderOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    if (origOp.output_buff().isa<mlir::BlockArgument>()) {
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.input());
    if (!isTrivial(inputShape)) {
        return mlir::failure();
    }
    const auto dstOrder = DimsOrder::fromValue(origOp.output_buff());
    const auto dstMaps = mlir::AffineMapAttr::get(dstOrder.toPermutationAffineMap(origOp.getContext()));
    rewriter.replaceOpWithNewOp<IERT::ImplicitReorderOp>(origOp, origOp.input(), dstMaps);
    return mlir::success();
}

}  // namespace

namespace {

#include <vpux/compiler/dialect/IERT/rewriters/generated/reorder.hpp.inc>

}  // namespace

void vpux::IERT::ReorderOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    populateWithGenerated(patterns);
    patterns.insert<ReorderToCopy>(context);
    patterns.insert<ConvertTrivialReorder>(context);
}

//
// fold
//

mlir::OpFoldResult vpux::IERT::ReorderOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (output_buff().isa<mlir::BlockArgument>()) {
        return nullptr;
    }

    if (input().getType() == output().getType()) {
        return input();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reorder(DimsOrder::fromValue(output()));
    }

    return nullptr;
}
