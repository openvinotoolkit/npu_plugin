//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AdaptiveAvgPoolUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::AdaptivePoolParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptivePoolParams});
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AdaptiveMaxPoolUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::AdaptivePoolParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptivePoolParams});
}
