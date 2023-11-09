//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MemPermuteOpAdaptor mem_permute(operands, attrs);
    if (mlir::failed(mem_permute.verify(loc))) {
        return mlir::failure();
    }

    inferPermuteReturnTypeComponents(mem_permute.input(), mem_permute.mem_perm(), mem_permute.dst_order(),
                                     inferredReturnShapes, false);

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
// FuseMemPermuteThroughConcat
//

class FuseMemPermuteThroughConcat final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermuteThroughConcat::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto concatOp = memPermuteOp.input().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<IE::MemPermuteOp> topMemPermutes;
    for (const auto& input : concatOp.inputs()) {
        auto topMemPermute = input.getDefiningOp<IE::MemPermuteOp>();
        if (topMemPermute == nullptr) {
            return mlir::failure();
        }
        topMemPermutes.push_back(topMemPermute);
    }

    IE::MemPermuteOp refMemPermute = topMemPermutes.front();
    SmallVector<mlir::Value> newConcatInputs;
    for (auto& topMemPermute : topMemPermutes) {
        if (refMemPermute.dst_order() != topMemPermute.dst_order()) {
            return mlir::failure();
        }
        if (refMemPermute.mem_perm() != topMemPermute.mem_perm()) {
            return mlir::failure();
        }
        newConcatInputs.push_back(topMemPermute.input());
    }

    const auto inputShape = getShape(refMemPermute.output());
    const auto inOrder = DimsOrder::fromValue(refMemPermute.output());
    const auto inMemShape = inOrder.toMemoryOrder(inputShape);
    const auto outputShape = getShape(concatOp.output());
    const auto outOrder = DimsOrder::fromValue(concatOp.output());
    const auto outMemShape = outOrder.toMemoryOrder(outputShape);
    const auto perm = refMemPermute.mem_perm();

    const auto permuteInOrder = DimsOrder::fromValue(refMemPermute.input());

    int32_t newAxis = -1;

    for (size_t idx = 0; idx < inMemShape.size(); ++idx) {
        if (inMemShape.raw()[idx] == outMemShape.raw()[idx]) {
            continue;
        }
        if (newAxis != -1) {
            // 2 axis concat
            return mlir::failure();
        }
        newAxis = perm.getDimPosition(static_cast<uint32_t>(idx));
        newAxis = permuteInOrder.dimAt(static_cast<size_t>(newAxis)).ind();
    }

    auto newConcat = rewriter.replaceOpWithNewOp<IE::ConcatOp>(concatOp, newConcatInputs, newAxis);

    auto prevMemPerm = refMemPermute.mem_perm();
    auto memPerm = memPermuteOp.mem_perm();
    auto newMemPerm = memPerm.compose(prevMemPerm);

    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(memPermuteOp, memPermuteOp.getType(), newConcat,
                                                  memPermuteOp.dst_orderAttr(), mlir::AffineMapAttr::get(newMemPerm));

    return mlir::success();
}

//
// FuseMemPermuteThroughExpand
//

/*  If we meet this pattern

    MemPermute(exchange dim_1 and dim_2)
        |
    Expand(expand on dim_1 or dim_2)
        |
    MemPermute(exchange dim_1 and dim_2)

    We can fuse the two MemPermute.
*/

bool canFuseThroughExpand(MemShapeRef firstInShape, mlir::AffineMap firstMemPerm, MemShapeRef secondInShape,
                          mlir::AffineMap secondMemPerm, IE::ExpandOp expandOp) {
    auto firstNonTrivialPerm = getPermutateDims(firstInShape, firstMemPerm);
    auto sceondNonTrivialPerm = getPermutateDims(secondInShape, secondMemPerm);

    if (firstNonTrivialPerm.size() != 2 || sceondNonTrivialPerm.size() != 2) {
        return false;
    }

    auto newMemPerm = secondMemPerm.compose(firstMemPerm);
    if (!isTrivialPermute(firstInShape, newMemPerm)) {
        return false;
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.pads_end());
    const auto expandInputShape = getShape(expandOp.input());
    const auto inOrder = DimsOrder::fromValue(expandOp.input());
    const auto expandInMemShape = inOrder.toMemoryOrder(expandInputShape);

    if (padsBegin.size() != expandInputShape.size() || padsEnd.size() != expandInputShape.size()) {
        return false;
    }

    for (auto ind : irange(expandInMemShape.size())) {
        const auto inDim = MemDim(inOrder.dimAt(ind).ind());
        if (expandInMemShape[inDim] == 1 && (padsBegin[ind] != 0 || padsEnd[ind] != 0)) {
            return false;
        }
    }

    return true;
}

mlir::ArrayAttr getNewPaddingAttr(mlir::MLIRContext* ctx, SmallVector<int64_t> pads, vpux::DimsOrder targetOrder) {
    SmallVector<int64_t> newPads(pads.size(), 0);

    for (auto ind : irange(pads.size())) {
        if (pads[ind] != 0) {
            const auto targetExpandIdx = MemDim(targetOrder.dimAt(ind).ind());
            newPads[targetExpandIdx.ind()] = pads[ind];
        }
    }

    return getIntArrayAttr(ctx, std::move(newPads));
}

class FuseMemPermuteThroughExpand final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermuteThroughExpand::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto expandOp = memPermuteOp.input().getDefiningOp<IE::ExpandOp>();
    if (expandOp == nullptr || !expandOp.output().hasOneUse()) {
        return mlir::failure();
    }

    auto topMemPermuteOp = expandOp.input().getDefiningOp<IE::MemPermuteOp>();
    if (topMemPermuteOp == nullptr || !topMemPermuteOp.output().hasOneUse()) {
        return mlir::failure();
    }

    auto inOrder = DimsOrder::fromValue(memPermuteOp.input());
    auto inShape = getShape(memPermuteOp.input());
    auto inMemShape = inOrder.toMemoryOrder(inShape);

    auto topInOrder = DimsOrder::fromValue(topMemPermuteOp.input());
    auto topInShape = getShape(topMemPermuteOp.input());
    auto topInMemShape = topInOrder.toMemoryOrder(topInShape);

    if (!canFuseThroughExpand(topInMemShape, topMemPermuteOp.mem_perm(), inMemShape, memPermuteOp.mem_perm(),
                              expandOp)) {
        return mlir::failure();
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.pads_end());

    // Get the real expanding axis
    const auto memPerm = DimsOrder::fromAffineMap(topMemPermuteOp.mem_perm());
    const auto targetOrder = vpux::applyPermutation(topInOrder, memPerm);

    const auto newPadsBeginAttr = getNewPaddingAttr(getContext(), padsBegin, targetOrder);
    const auto newPadsEndAttr = getNewPaddingAttr(getContext(), padsEnd, targetOrder);

    auto outputType = memPermuteOp.output().getType().cast<NDTypeInterface>();
    auto outputOrder = outputType.getDimsOrder();

    auto newExpandOp =
            rewriter.create<IE::ExpandOp>(expandOp.getLoc(), topMemPermuteOp.input(), newPadsBeginAttr, newPadsEndAttr);

    auto newMemPerm = memPermuteOp.mem_perm().compose(topMemPermuteOp.mem_perm());
    auto newPermuteCastOp = rewriter.create<IE::PermuteCastOp>(
            memPermuteOp.getLoc(), memPermuteOp.output().getType(), newExpandOp.output(),
            mlir::AffineMapAttr::get(outputOrder.toAffineMap(getContext())), mlir::AffineMapAttr::get(newMemPerm));

    memPermuteOp.replaceAllUsesWith(newPermuteCastOp.output());

    return mlir::success();
}

//
// FuseMemPermuteAndPermuteQuantize
//

class FuseMemPermuteAndPermuteQuantize final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<IE::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermuteAndPermuteQuantize::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto permuteQuantizeOp = memPermuteOp.input().getDefiningOp<IE::PermuteQuantizeOp>();
    if (permuteQuantizeOp == nullptr) {
        return mlir::failure();
    }
    // Can fuse MemPermute with PermuteQuantization in case only permutation (no quantization) is performed by this
    // PermuteQuantization Op.
    const auto permuteQuantizeOutElemType =
            permuteQuantizeOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (permuteQuantizeOutElemType.isa<mlir::quant::QuantizedType>()) {
        return mlir::failure();
    }

    return fusePermutations<IE::PermuteQuantizeOp, IE::MemPermuteOp>(memPermuteOp, rewriter);
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

void vpux::IE::MemPermuteOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<FuseMemPermutes>(context);
    patterns.add<ConvertToPermuteCast>(context);
    patterns.add<FusePermCastAndMemPerm>(context);
    patterns.add<FuseMemPermuteThroughConcat>(context);
    patterns.add<FuseMemPermuteThroughExpand>(context);
    patterns.add<FuseMemPermuteAndPermuteQuantize>(context);
}

mlir::OpFoldResult vpux::IE::MemPermuteOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType() && mem_perm().isIdentity()) {
        return input();
    }

    return nullptr;
}
