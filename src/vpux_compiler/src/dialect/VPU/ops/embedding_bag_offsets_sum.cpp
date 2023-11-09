//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EmbeddingBagOffsetsSumOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::EmbeddingBagOffsetsSumOpAdaptor embeddingBag(operands, attrs);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeEmbTable = embeddingBag.emb_table().getType().cast<vpux::NDTypeInterface>();
    auto embTableShape = to_small_vector(inTypeEmbTable.getShape().raw());
    SmallVector<int64_t> outShape(embTableShape);

    if (embeddingBag.offsets() != nullptr) {
        const auto inTypeOffsets = embeddingBag.offsets().getType().cast<vpux::NDTypeInterface>();
        SmallVector<int64_t> offsetsOutShape(to_small_vector(inTypeOffsets.getShape().raw()));
        outShape[0] = offsetsOutShape[0];
    } else if (embeddingBag.offsets_value().has_value()) {
        const auto offsetsAttr = parseIntArrayAttr<int32_t>(embeddingBag.offsets_value().value());
        outShape[0] = offsetsAttr.size();
    } else {
        return errorAt(loc, "Offsets input was not provided properly");
    }

    const auto outType = inTypeEmbTable.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::EmbeddingBagOffsetsSumOp::serialize(EMU::BlobWriter& writer) {
    const auto indices = writer.createVector(parseIntArrayAttr<int32_t>(indices_value().value()));
    const auto offsets = writer.createVector(parseIntArrayAttr<int32_t>(offsets_value().value()));

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    EMU::BlobWriter::Vector<uint16_t> serializedWeights;
    const auto weightsArr = parseFPArrayAttr<double>(per_sample_weights_value().value());
    serializedWeights = getVecFP16(weightsArr);

    MVCNN::EmbeddingBagOffsetsSumParamsBuilder builder(writer);
    builder.add_indices(indices);
    builder.add_offsets(offsets);
    builder.add_default_index(checked_cast<int32_t>(default_index_value()));
    builder.add_weights(serializedWeights);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this,
                                     {paramsOff.Union(), MVCNN::SoftwareLayerParams_EmbeddingBagOffsetsSumParams});
}
