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

namespace {
// This method converts value from AdaptivePoolMode view to corresponds t_AdaptivePool_mode view from runtime
MVCNN::AdaptivePoolMode AdaptivePoolMode2Int32(IE::AdaptivePoolMode mode) {
    MVCNN::AdaptivePoolMode out_code = MVCNN::AdaptivePoolMode::AdaptivePoolMode_AVG;
    switch (mode) {
    case IE::AdaptivePoolMode::avg:
        out_code = MVCNN::AdaptivePoolMode::AdaptivePoolMode_AVG;
        break;
    case IE::AdaptivePoolMode::max:
        out_code = MVCNN::AdaptivePoolMode::AdaptivePoolMode_MAX;
        break;
    default:
        VPUX_THROW("Unknown AdaptivePoolMode, avg and max modes are supported only");
    }
    return out_code;
}
}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(AdaptivePoolUPAOp op) {
    const auto inShapeFeatureMap = getShape(op.input());
    const auto inShapeCoord = getShape(op.coords());

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(op, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(op, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    const auto output_dim = op.output_dim();

    if (output_dim <= 0) {
        return errorAt(op, "Attribute output_dim should be positive.");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::AdaptivePoolUPAOp::serialize(VPUIP::BlobWriter& writer) {

    const auto _mode = mode().hasValue() ? mode().getValue() : IE::AdaptivePoolMode::avg;

    MVCNN::AdaptivePoolParamsBuilder builder(writer);

    builder.add_mode(AdaptivePoolMode2Int32(_mode));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptivePool});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseAdaptivePool(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                            ArrayRef<mlir::Value> outputs,
                                                            const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAAdaptivePool supports only 2 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAAdaptivePool supports only 1 output, got {0}", outputs.size());

    IE::AdaptivePoolMode mode;
    switch (params->mode()) {
    case 0:
        mode = IE::AdaptivePoolMode::avg;
        break;
    case 1:
        mode = IE::AdaptivePoolMode::max;
        break;
    default:
        VPUX_THROW("Unknown AdaptivePoolMode. avg and max mode are supported only");
    }

    return builder.create<VPUIP::AdaptivePoolUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                   IE::AdaptivePoolModeAttr::get(_ctx, mode));
}