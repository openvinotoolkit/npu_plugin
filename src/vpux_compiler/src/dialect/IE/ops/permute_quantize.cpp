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

mlir::LogicalResult vpux::IE::PermuteQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PermuteQuantizeOpAdaptor permute_quantize(operands, attrs);
    if (mlir::failed(permute_quantize.verify(loc))) {
        return mlir::failure();
    }

    mlir::Value input = permute_quantize.input();
    mlir::AffineMap memPerm = permute_quantize.mem_perm();
    mlir::AffineMap dstOrder = permute_quantize.dst_order();
    const auto dstElemType = permute_quantize.dstElemType();

    const auto padBegin = parseIntArrayAttr<int64_t>(permute_quantize.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(permute_quantize.pads_end());

    const auto inType = permute_quantize.input().getType().cast<vpux::NDTypeInterface>();

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dstOrder);

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    const auto inShapeExpanded = newType.getShape();

    const auto inMemShape = inOrder.toMemoryOrder(inShapeExpanded);
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    const auto outDesc = vpux::getTensorAttr(dstOrder, nullptr);

    inferredReturnShapes.emplace_back(outShape.raw(), dstElemType, outDesc);

    return mlir::success();
}

namespace {

//
// ConvertToPermuteCast
//

class ConvertToPermuteCast final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    using mlir::OpRewritePattern<IE::PermuteQuantizeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto inShape = getShape(origOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    const auto inputType = origOp.input().getType().cast<NDTypeInterface>().getElementType();
    const auto outputType = origOp.output().getType().cast<NDTypeInterface>().getElementType();

    if (!isTrivialPermute(inMemShape, origOp.mem_perm()) || inputType != outputType ||
        inShape != getShape(origOp.output())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(origOp, origOp.input(), origOp.dst_orderAttr(),
                                                   origOp.mem_permAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::PermuteQuantizeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* context) {
    patterns.add<ConvertToPermuteCast>(context);
}

mlir::OpFoldResult vpux::IE::PermuteQuantizeOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType() && mem_perm().isIdentity()) {
        return input();
    }

    return nullptr;
}
