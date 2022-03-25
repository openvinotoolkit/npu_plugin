//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::MVNUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::MVNParamsBuilder builder(writer);
    builder.add_across_channels(across_channels().getValueOr(false));
    builder.add_normalize_variance(normalize_variance().getValueOr(false));
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_MVNParams});
}

mlir::LogicalResult vpux::VPUIP::verifyOp(MVNUPAOp op) {
    const auto inShape = getShape(op.input());

    if (inShape.size() != 3 && inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(op, "Input shape should have 3, 4 or 5 dimensions");
    }

    return mlir::success();
}
