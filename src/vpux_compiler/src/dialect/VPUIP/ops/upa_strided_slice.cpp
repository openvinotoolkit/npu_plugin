//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::StridedSliceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    auto attrToVector = [&](mlir::ArrayAttr attr) {
        return to_std_vector(parseIntArrayAttr<uint32_t>(attr));
    };

    const auto beginsVec = attrToVector(begins());
    const auto endsVec = attrToVector(ends());
    const auto stridesVec = attrToVector(strides());

    const auto paramsOff = MVCNN::CreateStridedSliceParamsDirect(writer, &beginsVec, &endsVec, &stridesVec);

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_StridedSliceParams});
}
