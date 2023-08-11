//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::BucketizeUPAOp::verify() {
    const mlir::Type constInt32 = getSInt32Type(getContext());

    if (!(output_type() == constInt32)) {
        return errorAt(*this, "Attribute output_type should have only SI32 type. Got {0} type", output_type());
    }
    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::BucketizeUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::BucketizeParamsBuilder builder(writer);

    MVCNN::BucketizeOutputType outType;

    if (this->output_type().isSignedInteger(32)) {
        outType = MVCNN::BucketizeOutputType::BucketizeOutputType_I32;
    } else if (this->output_type().isSignedInteger(64)) {
        outType = MVCNN::BucketizeOutputType::BucketizeOutputType_I64;
    } else {
        VPUX_THROW("Unsupported type for output_type attribute. Got {0} instead of SI32 or SI64 type, that is defined "
                   "in schema for output_type.",
                   this->output_type());
    }

    builder.add_output_type(outType);
    builder.add_with_right_bound(with_right_bound());

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_BucketizeParams});
}
