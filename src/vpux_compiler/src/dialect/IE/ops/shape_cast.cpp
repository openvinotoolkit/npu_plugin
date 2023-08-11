//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace IE;

mlir::LogicalResult vpux::IE::ShapeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ShapeCastOpAdaptor shapeCast(operands, attrs);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.shape());

    const auto inType = shapeCast.source().getType().cast<vpux::NDTypeInterface>();
    const auto outDesc = IE::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
    inferredReturnTypes.emplace_back(outShape, inType.getElementType(), outDesc);
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::ShapeCastOp::fold(ArrayRef<mlir::Attribute>) {
    if (source().getType() == result().getType()) {
        return source();
    }

    return nullptr;
}

//
// FuseShapeCast
//

namespace {
class FuseShapeCast final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    using mlir::OpRewritePattern<IE::ShapeCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseShapeCast::matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.source().getDefiningOp<IE::ShapeCastOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(origOp, prevOp.source(), origOp.shape());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ShapeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseShapeCast>(ctx);
}
