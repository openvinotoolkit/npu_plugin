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

    const auto inTypeEmbTable = embeddingBag.input().getType().cast<vpux::NDTypeInterface>();
    const auto inTypeOffsets = parseIntArrayAttr<int32_t>(embeddingBag.offsets_value());
    const auto inShape = inTypeEmbTable.getShape().raw();

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inShape.size(); i++)
        outShape.emplace_back(inShape[i]);
    outShape[0] = inTypeOffsets.size();

    const auto outType = inTypeEmbTable.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::EmbeddingBagOffsetsSumOp::serialize(EMU::BlobWriter& writer) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    EMU::BlobWriter::Vector<uint16_t> serializedWeights;
    const auto weightsArr = parseFPArrayAttr<double>(weights_value());
    const auto indices = writer.createVector(parseIntArrayAttr<int32_t>(indices_value()));
    const auto offsets = writer.createVector(parseIntArrayAttr<int32_t>(offsets_value()));
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
