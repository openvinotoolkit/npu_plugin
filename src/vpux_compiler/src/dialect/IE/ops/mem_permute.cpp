//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::MemPermuteOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MemPermuteOpAdaptor mem_permute(operands, attrs);
    if (mlir::failed(mem_permute.verify(loc))) {
        return mlir::failure();
    }

    inferPermuteReturnTypeComponents(mem_permute.input(), mem_permute.mem_perm().getValue(),
                                     mem_permute.dst_order().getValue(), inferredReturnShapes, false);

    return mlir::success();
}

namespace {

//
// FuseMemPermutes
//

class FuseMemPermutes final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermutes::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::MemPermuteOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
}

//
// FusePermCastAndMemPerm
//

// PermuteCast -> MemPermute ===> MemPermute

class FusePermCastAndMemPerm final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FusePermCastAndMemPerm::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                            mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::PermuteCastOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
}

//
// ConvertToPermuteCast
//

class ConvertToPermuteCast final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto inOrder = DimsOrder::fromValue(memPermuteOp.input());
    const auto inShape = getShape(memPermuteOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    if (!isTrivialPermute(inMemShape, memPermuteOp.mem_perm())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(memPermuteOp, memPermuteOp.input(), memPermuteOp.dst_orderAttr(),
                                                   memPermuteOp.mem_permAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::MemPermuteOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.insert<FuseMemPermutes>(context);
    patterns.insert<ConvertToPermuteCast>(context);
    patterns.insert<FusePermCastAndMemPerm>(context);
}

mlir::OpFoldResult vpux::IE::MemPermuteOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType() && mem_perm().isIdentity()) {
        return input();
    }

    return nullptr;
}
