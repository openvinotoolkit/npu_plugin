//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<int64_t> extractNumSegments(mlir::Location loc,
                                            IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum) {
    if (embeddingSegmentsSum.num_segments() != nullptr) {
        auto numSegmentsConst = embeddingSegmentsSum.num_segments().getDefiningOp<Const::DeclareOp>();
        if (numSegmentsConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for numSegments");
        }

        const auto numSegmentsContent = numSegmentsConst.content();
        if (!numSegmentsContent.isSplat()) {
            return errorAt(loc, "numSegments value must be a scalar");
        }

        int64_t numSegments = numSegmentsContent.getSplatValue<int64_t>();
        return numSegments;
    } else if (embeddingSegmentsSum.num_segments_value().hasValue()) {
        return embeddingSegmentsSum.num_segments_value().getValue();
    }
    return errorAt(loc, "NumSegments was not provided");
}

}  // namespace

mlir::LogicalResult vpux::IE::EmbeddingSegmentsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum(operands, attrs);
    if (mlir::failed(embeddingSegmentsSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = embeddingSegmentsSum.emb_table().getType().cast<mlir::ShapedType>();

    auto embTableShape = to_small_vector(inType.getShape());

    const auto numSegments = extractNumSegments(loc, embeddingSegmentsSum);
    if (mlir::failed(numSegments)) {
        return mlir::failure();
    }

    int64_t numSegmentsVal = checked_cast<int64_t>(*numSegments);

    SmallVector<int64_t> outShape(embTableShape);
    outShape[0] = numSegmentsVal;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp EmbeddingSegmentsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::EmbeddingSegmentsSumOp embeddingSegmentsSumOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto indicesAttr = vpux::IE::getIntArrayAttrValue(embeddingSegmentsSumOp.indices());
    auto segmentIdsAttr = vpux::IE::getIntArrayAttrValue(embeddingSegmentsSumOp.segment_ids());
    auto numSegmentsAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.num_segments(), rewriter);
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.default_index(), rewriter);
    auto perSampleWeightsAttr = vpux::IE::getFloatArrayAttrValue(embeddingSegmentsSumOp.per_sample_weights());

    if ((embeddingSegmentsSumOp.default_index_valueAttr() == nullptr) && (defaultIndexAttr == nullptr)) {
        int32_t defaultValueDefaultIndex =
                -1;  // The OpenVINO spec expects default value 0. However, the ACT Shave kernel implementation fills
                     // the empty segments with zero when a negative value is provided.
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    if ((embeddingSegmentsSumOp.per_sample_weights_valueAttr() == nullptr) && (perSampleWeightsAttr == nullptr)) {
        SmallVector<float> defaultValuePerSampleWeights(indicesAttr.size(), 1);
        perSampleWeightsAttr = getFPArrayAttr(embeddingSegmentsSumOp.getContext(), defaultValuePerSampleWeights);
    }

    if ((indicesAttr == nullptr) && (segmentIdsAttr == nullptr) && (numSegmentsAttr == nullptr) &&
        (defaultIndexAttr == nullptr) && (perSampleWeightsAttr == nullptr)) {
        return mlir::failure();
    }

    const auto indices = (indicesAttr == nullptr) ? embeddingSegmentsSumOp.indices() : nullptr;
    const auto segmentIds = (segmentIdsAttr == nullptr) ? embeddingSegmentsSumOp.segment_ids() : nullptr;
    const auto numSegments = (numSegmentsAttr == nullptr) ? embeddingSegmentsSumOp.num_segments() : nullptr;
    const auto defaultIndex = (defaultIndexAttr == nullptr) ? embeddingSegmentsSumOp.default_index() : nullptr;
    const auto perSampleWeights =
            (perSampleWeightsAttr == nullptr) ? embeddingSegmentsSumOp.per_sample_weights() : nullptr;

    rewriter.replaceOpWithNewOp<IE::EmbeddingSegmentsSumOp>(
            embeddingSegmentsSumOp, embeddingSegmentsSumOp.getType(), embeddingSegmentsSumOp.emb_table(), indices,
            segmentIds, numSegments, defaultIndex, perSampleWeights, indicesAttr, segmentIdsAttr, numSegmentsAttr,
            defaultIndexAttr, perSampleWeightsAttr);

    return mlir::success();
}

}  // namespace

void vpux::IE::EmbeddingSegmentsSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                   mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
