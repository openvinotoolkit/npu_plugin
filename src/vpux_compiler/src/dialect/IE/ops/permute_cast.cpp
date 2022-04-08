///
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::PermuteCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PermuteCastOpAdaptor permuteCast(operands, attrs);
    if (mlir::failed(permuteCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inOrder = DimsOrder::fromValue(permuteCast.input());
    const auto inShape = getShape(permuteCast.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (!isTrivialPermute(inMemShape, permuteCast.mem_perm().getValue())) {
        return errorAt(loc, "Operation represents non trivial permutation");
    }

    inferPermuteReturnTypeComponents(permuteCast.input(), permuteCast.mem_perm().getValue(),
                                     permuteCast.dst_order().getValue(), inferredReturnShapes, true);

    return mlir::success();
}

namespace {

//
// FusePermuteCasts
//

class FusePermuteCasts final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<IE::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp permuteCastOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FusePermuteCasts::matchAndRewrite(IE::PermuteCastOp permuteCastOp,
                                                      mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::PermuteCastOp, IE::PermuteCastOp>(permuteCastOp, rewriter);
}

//
// FuseMemPermAndPermCast
//

// MemPermute -> PermuteCast ===> MemPermute

class FuseMemPermAndPermCast final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<IE::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp permuteCastOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermAndPermCast::matchAndRewrite(IE::PermuteCastOp permuteCastOp,
                                                            mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::MemPermuteOp, IE::PermuteCastOp>(permuteCastOp, rewriter);
}

}  // namespace

void vpux::IE::PermuteCastOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.insert<FusePermuteCasts>(context);
    patterns.insert<FuseMemPermAndPermCast>(context);
}

mlir::OpFoldResult vpux::IE::PermuteCastOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType() && mem_perm().isIdentity()) {
        return input();
    }

    return nullptr;
}
