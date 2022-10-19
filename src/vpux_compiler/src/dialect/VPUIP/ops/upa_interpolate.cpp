//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/utils/core/enums.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::InterpolateUPAOp::inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
    auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Interpolate input shape expected to have 4 dimensions, but has {0}",
                      inputShape.size());

    // Select NCHW layout due to performance reasons
    // [Track number: E#25302]
    auto channels = inputShape[1];
    const auto antialias = mlir::cast<IE::InterpolateOp>(op).attr().antialias().getValue();
    if (channels == 1 || antialias) {
        VPUIP::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    } else {
        VPUIP::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::NHWC});
    }
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::InterpolateUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::InterpolateParamsBuilder builder(writer);

    const auto interpolateModeIter = VPUIP::supportedInterpModeMap.find(mode());
    VPUX_THROW_UNLESS(interpolateModeIter != VPUIP::supportedInterpModeMap.end(), "Unsupported interpolate mode {0}",
                      mode());
    builder.add_interpolationMode(interpolateModeIter->second);

    const auto coordModeIter = VPUIP::coordTransformModeMap.find(coord_mode());
    VPUX_THROW_UNLESS(coordModeIter != VPUIP::coordTransformModeMap.end(),
                      "Unsupported coordinate transformation mode {0}", coord_mode());
    builder.add_coordTransformMode(coordModeIter->second);

    const auto nearestModeIter = VPUIP::nearestModeMap.find(nearest_mode());
    VPUX_THROW_UNLESS(nearestModeIter != VPUIP::nearestModeMap.end(), "Unsupported nearest mode {0}", nearest_mode());
    builder.add_nearestMode(nearestModeIter->second);

    builder.add_align_corners(coord_mode() == IE::InterpolateCoordMode::ALIGN_CORNERS);
    builder.add_antialias(antialias());

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_InterpolateParams});
}
