//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

mlir::LogicalResult vpux::VPUIP::verifyOp(AdaptivePoolUPAOp op) {
    const auto inputShape = getShape(op.input());
    const auto pooledSpatialShape = getShape(op.pooled_spatial_shape());

    if (inputShape.size() != 3 && inputShape.size() != 4 && inputShape.size() != 5) {
        return errorAt(op, "Input shape should have 3, 4 or 5 dimensions");
    }
    if (pooledSpatialShape.size() != 1) {
        return errorAt(op, "Dimension of input2 should be 1. Got {0} D tensor", pooledSpatialShape.size());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AdaptivePoolUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::AdaptiveAvgPoolParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptiveAvgPoolParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseAdaptiveAvgPool(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask*) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAAdaptivePool supports only 2 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAAdaptivePool supports only 1 output, got {0}", outputs.size());

    return builder.create<VPUIP::AdaptivePoolUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0]);
}
