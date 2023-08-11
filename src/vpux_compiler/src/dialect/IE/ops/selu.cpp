//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::SeluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SeluOpAdaptor selu(operands, attrs);
    if (mlir::failed(selu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = selu.data().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SeluOp> {
public:
    using mlir::OpRewritePattern<IE::SeluOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SeluOp seluOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult getAttrValue(mlir::Value attr, float& attrValue) {
    if (attr == nullptr) {
        return mlir::failure();
    }
    auto attrOp = attr.getDefiningOp<Const::DeclareOp>();
    if (attrOp == nullptr) {
        return mlir::failure();
    }

    const auto attrContent = attrOp.content();
    if (!attrContent.isSplat()) {
        return mlir::failure();
    }

    attrValue = attrContent.getSplatValue<float>();

    return mlir::success();
}

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SeluOp seluOp, mlir::PatternRewriter& rewriter) const {
    if ((seluOp.alphaValue()) || (seluOp.lambdaValue())) {
        return mlir::failure();
    }

    float alphaValue;
    float lambdaValue;

    if (getAttrValue(seluOp.alpha(), alphaValue).failed()) {
        return mlir::failure();
    }
    if (getAttrValue(seluOp.lambda(), lambdaValue).failed()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::SeluOp>(seluOp, seluOp.getType(), seluOp.data(), nullptr, nullptr,
                                            rewriter.getF64FloatAttr(alphaValue),
                                            rewriter.getF64FloatAttr(lambdaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::SeluOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
