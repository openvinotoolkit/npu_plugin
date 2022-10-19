//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::RegionYoloUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::Vector<int32_t> serializedMask;
    if (mask().hasValue()) {
        serializedMask = writer.createVector(parseIntArrayAttr<int32_t>(mask().getValue()));
    }

    MVCNN::RegionYOLOParamsBuilder builder(writer);
    builder.add_coords(checked_cast<int32_t>(coords()));
    builder.add_classes(checked_cast<int32_t>(classes()));
    builder.add_num(checked_cast<int32_t>(regions()));
    builder.add_do_softmax(do_softmax().getValueOr(false));
    if (mask().hasValue()) {
        builder.add_mask(serializedMask);
    }

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_RegionYOLOParams});
}

void vpux::VPUIP::RegionYoloUPAOp::inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) {
    auto regionYoloOp = mlir::dyn_cast<IE::RegionYoloOp>(origOp);
    VPUX_THROW_UNLESS(regionYoloOp != nullptr, "Operation '{0}' is not a RegionYolo", origOp->getName());

    if (regionYoloOp.do_softmax()) {
        IE::fillDefaultLayoutInfo(info);
    } else {
        VPUIP::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    }
}
