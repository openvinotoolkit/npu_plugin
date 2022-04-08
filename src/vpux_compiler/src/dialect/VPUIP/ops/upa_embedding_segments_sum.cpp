//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EmbeddingSegmentsSumUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto indices = writer.createVector(parseIntArrayAttr<int32_t>(indices_value()));
    const auto segmentIds = writer.createVector(parseIntArrayAttr<int32_t>(segment_ids_value()));

    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = float16(val);
        return valFP16.to_bits();
    };
    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    VPUIP::BlobWriter::Vector<uint16_t> serializedWeights;
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
