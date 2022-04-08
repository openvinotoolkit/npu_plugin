//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::EmbeddingBagOffsetsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));
    IE::EmbeddingBagOffsetsSumOpAdaptor embeddingBag(operands, attrs);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeEmbTable = embeddingBag.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inTypeEmbTable.getShape();

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inShape.size(); i++)
        outShape.emplace_back(inShape[i]);

    if (embeddingBag.offsets() != nullptr) {
        const auto inTypeOffsets = embeddingBag.offsets().getType().cast<mlir::ShapedType>();
        const auto offsetsShape = inTypeOffsets.getShape();
        outShape[0] = offsetsShape.size();
    } else if (embeddingBag.offsets_value().hasValue()) {
        const auto offsetsAttr = parseIntArrayAttr<int32_t>(embeddingBag.offsets_value().getValue());
        outShape[0] = offsetsAttr.size();
    } else
        return errorAt(loc, "Offsets input was not provided properly");

    inferredReturnShapes.emplace_back(outShape, inTypeEmbTable.getElementType());
    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp EmbeddingBagOffsetsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp embeddingBagOffsetsSumOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto indicesAttr = vpux::IE::getIntArrayAttrValue(embeddingBagOffsetsSumOp.indices());
    auto offsetsAttr = vpux::IE::getIntArrayAttrValue(embeddingBagOffsetsSumOp.offsets());
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingBagOffsetsSumOp.default_index(), rewriter);
    auto weightsAttr = vpux::IE::getFloatArrayAttrValue(embeddingBagOffsetsSumOp.weights());
    if ((embeddingBagOffsetsSumOp.default_index_valueAttr() == nullptr) && (defaultIndexAttr == nullptr)) {
        int32_t defaultValueDefaultIndex = 0;
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }
    if ((embeddingBagOffsetsSumOp.weights_valueAttr() == nullptr) && (weightsAttr == nullptr)) {
        SmallVector<float> defaultValueWeights(indicesAttr.size(), 1);
        weightsAttr = getFPArrayAttr(embeddingBagOffsetsSumOp.getContext(), defaultValueWeights);
    }
    if ((indicesAttr == nullptr) && (offsetsAttr == nullptr) && (defaultIndexAttr == nullptr) &&
        (weightsAttr == nullptr)) {
        return mlir::failure();
    }
    const auto indices = (indicesAttr == nullptr) ? embeddingBagOffsetsSumOp.indices() : nullptr;
    const auto offsets = (offsetsAttr == nullptr) ? embeddingBagOffsetsSumOp.offsets() : nullptr;
    const auto weights = (weightsAttr == nullptr) ? embeddingBagOffsetsSumOp.weights() : nullptr;
    const auto defaultIndex = (defaultIndexAttr == nullptr) ? embeddingBagOffsetsSumOp.default_index() : nullptr;
    rewriter.replaceOpWithNewOp<IE::EmbeddingBagOffsetsSumOp>(
            embeddingBagOffsetsSumOp, embeddingBagOffsetsSumOp.getType(), embeddingBagOffsetsSumOp.input(), indices,
            offsets, defaultIndex, weights, indicesAttr, offsetsAttr, defaultIndexAttr, weightsAttr);
    return mlir::success();
}

}  // namespace

void vpux::IE::EmbeddingBagOffsetsSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                     mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
