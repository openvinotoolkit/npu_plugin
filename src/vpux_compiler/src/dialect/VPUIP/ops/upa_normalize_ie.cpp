//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NormalizeIEUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::NormalizeParamsBuilder builder(writer);
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    builder.add_across_spatial(static_cast<int32_t>(across_spatial()));
    builder.add_channel_shared(static_cast<int32_t>(channel_shared()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormalizeParams});
}
