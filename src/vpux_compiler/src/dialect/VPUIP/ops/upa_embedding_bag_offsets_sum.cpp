//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EmbeddingBagOffsetsSumUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    VPUIP::BlobWriter::Vector<uint16_t> serializedWeights;
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
