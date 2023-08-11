//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CumSumUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CumSumParamsBuilder builder(writer);
    builder.add_exclusive(checked_cast<bool>(exclusive().value_or(false)));
    builder.add_reverse(checked_cast<bool>(reverse().value_or(false)));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CumSumParams});
}
