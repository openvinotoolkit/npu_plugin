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

//
// verify
//

mlir::LogicalResult vpux::IE::EmbeddingBagOffsetsSumOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = tensor.getType().cast<vpux::NDTypeInterface>().getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(default_index())) {
        return errorAt(*this, "default_index should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::EmbeddingBagOffsetsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::EmbeddingBagOffsetsSumOpAdaptor embeddingBag(operands, attrs);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeEmbTable = embeddingBag.emb_table().getType().cast<mlir::ShapedType>();

    auto embTableShape = to_small_vector(inTypeEmbTable.getShape());
    SmallVector<int64_t> outShape(embTableShape);

    if (embeddingBag.offsets() != nullptr) {
        const auto inTypeOffsets = embeddingBag.offsets().getType().cast<mlir::ShapedType>();
        const auto offsetsShape = inTypeOffsets.getShape();
        outShape[0] = checked_cast<int64_t>(offsetsShape[0]);
    } else if (embeddingBag.offsets_value().has_value()) {
        const auto offsetsAttr = parseIntArrayAttr<int32_t>(embeddingBag.offsets_value().value());
        outShape[0] = offsetsAttr.size();
    } else
        return errorAt(loc, "Offsets input was not provided properly");

    inferredReturnShapes.emplace_back(outShape, inTypeEmbTable.getElementType());
    return mlir::success();
}

//
// ConvertConstToAttrVPUX30XX
//

namespace {
class ConvertConstToAttrVPUX30XX final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp EmbeddingBagOffsetsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttrVPUX30XX::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp embeddingBagOffsetsSumOp,
                                                                mlir::PatternRewriter& rewriter) const {
    const auto arch = VPU::getArch(embeddingBagOffsetsSumOp);
    if (arch != VPU::ArchKind::VPUX30XX) {
        return mlir::failure();
    }

    if ((embeddingBagOffsetsSumOp.indices_valueAttr() != nullptr) &&
        (embeddingBagOffsetsSumOp.offsets_valueAttr() != nullptr) &&
        (embeddingBagOffsetsSumOp.default_index_valueAttr() != nullptr) &&
        (embeddingBagOffsetsSumOp.per_sample_weights_valueAttr() != nullptr)) {
        return mlir::failure();
    }

    auto indicesAttr = vpux::IE::getIntArrayAttrValue(embeddingBagOffsetsSumOp.indices());
    auto offsetsAttr = vpux::IE::getIntArrayAttrValue(embeddingBagOffsetsSumOp.offsets());
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingBagOffsetsSumOp.default_index(), rewriter);
    auto perSampleWeightsAttr = vpux::IE::getFloatArrayAttrValue(embeddingBagOffsetsSumOp.per_sample_weights());

    if (defaultIndexAttr == nullptr) {
        // The OpenVINO spec expects default value 0. However, the ACT Shave kernel implementation
        // fills the empty segments with zero when a negative value is provided.
        int32_t defaultValueDefaultIndex = -1;
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    if ((embeddingBagOffsetsSumOp.per_sample_weights_valueAttr() == nullptr) && (perSampleWeightsAttr == nullptr)) {
        SmallVector<float> defaultValuePerSampleWeights(indicesAttr.size(), 1);
        perSampleWeightsAttr = getFPArrayAttr(embeddingBagOffsetsSumOp.getContext(), defaultValuePerSampleWeights);
    }

    if ((indicesAttr == nullptr) && (offsetsAttr == nullptr) && (defaultIndexAttr == nullptr) &&
        (perSampleWeightsAttr == nullptr)) {
        return mlir::failure();
    }

    const auto indices = (indicesAttr == nullptr) ? embeddingBagOffsetsSumOp.indices() : nullptr;
    const auto offsets = (offsetsAttr == nullptr) ? embeddingBagOffsetsSumOp.offsets() : nullptr;
    const auto defaultIndex = (defaultIndexAttr == nullptr) ? embeddingBagOffsetsSumOp.default_index() : nullptr;
    const auto perSampleWeights =
            (perSampleWeightsAttr == nullptr) ? embeddingBagOffsetsSumOp.per_sample_weights() : nullptr;

    rewriter.replaceOpWithNewOp<IE::EmbeddingBagOffsetsSumOp>(
            embeddingBagOffsetsSumOp, embeddingBagOffsetsSumOp.getType(), embeddingBagOffsetsSumOp.emb_table(), indices,
            offsets, defaultIndex, perSampleWeights, indicesAttr, offsetsAttr, defaultIndexAttr, perSampleWeightsAttr);

    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttrVPUX37XX
//

namespace {

class ConvertConstToAttrVPUX37XX final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp EmbeddingBagOffsetsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttrVPUX37XX::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp embeddingBagOffsetsSumOp,
                                                                mlir::PatternRewriter& rewriter) const {
    const auto arch = VPU::getArch(embeddingBagOffsetsSumOp);
    if (arch != VPU::ArchKind::VPUX37XX) {
        return mlir::failure();
    }

    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingBagOffsetsSumOp.default_index(), rewriter);

    if ((embeddingBagOffsetsSumOp.default_index_valueAttr() == nullptr) && (defaultIndexAttr == nullptr)) {
        int32_t defaultValueDefaultIndex = -1;
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    if (defaultIndexAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::EmbeddingBagOffsetsSumOp>(
            embeddingBagOffsetsSumOp, embeddingBagOffsetsSumOp.getType(), embeddingBagOffsetsSumOp.emb_table(),
            embeddingBagOffsetsSumOp.indices(), embeddingBagOffsetsSumOp.offsets(), nullptr /*defaultIndex*/,
            embeddingBagOffsetsSumOp.per_sample_weights(), nullptr /*indicesAttr*/, nullptr /*offsetsAttr*/,
            defaultIndexAttr, nullptr);

    return mlir::success();
}

}  // namespace

void vpux::IE::EmbeddingBagOffsetsSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                     mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttrVPUX30XX>(context);
    patterns.add<ConvertConstToAttrVPUX37XX>(context);
}
