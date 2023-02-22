//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::EmbeddingSegmentsSumOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum(operands, attrs);
    if (mlir::failed(embeddingSegmentsSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = embeddingSegmentsSum.emb_table().getType().cast<vpux::NDTypeInterface>();

    auto embTableShape = to_small_vector(inType.getShape().raw());

    SmallVector<int64_t> outShape(embTableShape);
    outShape[0] = checked_cast<int64_t>(embeddingSegmentsSum.num_segments_value());

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::EmbeddingSegmentsSumOp::serialize(EMU::BlobWriter& writer) {
    const auto indices = writer.createVector(parseIntArrayAttr<int32_t>(indices_value()));
    const auto segmentIds = writer.createVector(parseIntArrayAttr<int32_t>(segment_ids_value()));

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };
    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    EMU::BlobWriter::Vector<uint16_t> serializedWeights;
    const auto weightsArr = parseFPArrayAttr<double>(per_sample_weights_value());
    serializedWeights = getVecFP16(weightsArr);

    MVCNN::EmbeddingSegmentsSumParamsBuilder builder(writer);

    builder.add_indices(indices);
    builder.add_segment_ids(segmentIds);
    builder.add_num_segments(checked_cast<int32_t>(num_segments_value()));
    builder.add_default_index(checked_cast<int32_t>(default_index_value()));
    builder.add_per_sample_weights(serializedWeights);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EmbeddingSegmentsSumParams});
}
