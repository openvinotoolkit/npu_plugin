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
// This method converts value from PSROIPoolingMode view to corresponds t_PSROIPooling_mode view from runtime
MVCNN::PSROIPoolingMode PSROIPoolingMode2Int32(IE::PSROIPoolingMode mode) {
    MVCNN::PSROIPoolingMode out_code = MVCNN::PSROIPoolingMode::PSROIPoolingMode_AVERAGE;
    switch (mode) {
    case IE::PSROIPoolingMode::average:
        out_code = MVCNN::PSROIPoolingMode::PSROIPoolingMode_AVERAGE;
        break;
    case IE::PSROIPoolingMode::bilinear:
        out_code = MVCNN::PSROIPoolingMode::PSROIPoolingMode_BILINEAR;
        break;
    default:
        VPUX_THROW("Unknown PSROIPoolingMode, average and bilinear modes are supported only");
    }
    return out_code;
}
}  // namespace

mlir::LogicalResult vpux::VPUIP::verifyOp(PSROIPoolingUPAOp op) {
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

void vpux::VPUIP::PSROIPoolingUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                         mlir::Value input, mlir::Value coords, mlir::Value output,
                                         mlir::IntegerAttr output_dim, mlir::FloatAttr spatial_scale,
                                         mlir::IntegerAttr group_size, mlir::IntegerAttr spatial_bins_x,
                                         mlir::IntegerAttr spatial_bins_y, IE::PSROIPoolingModeAttr mode) {
    build(odsBuilder, odsState,
          input, coords, output, output_dim,
          spatial_scale, group_size, spatial_bins_x, spatial_bins_y, mode, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PSROIPoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto _group_size = group_size().hasValue() ? group_size().getValue() : 1;
    const auto _spatial_bins_x = spatial_bins_x().hasValue() ? spatial_bins_x().getValue() : 1;
    const auto _spatial_bins_y = spatial_bins_y().hasValue() ? spatial_bins_y().getValue() : 1;
    const auto _mode = mode().hasValue() ? mode().getValue() : IE::PSROIPoolingMode::average;

    MVCNN::PSROIPoolingParamsBuilder builder(writer);

    builder.add_output_dim   (checked_cast<uint32_t> (output_dim()));
    builder.add_spatial_scale(checked_cast<float>   (spatial_scaleAttr().getValueAsDouble()));
    builder.add_group_size   (checked_cast<uint32_t> (_group_size));
    builder.add_pooled_w     (checked_cast<uint32_t> (_group_size));
    builder.add_pooled_h     (checked_cast<uint32_t> (_group_size));
    builder.add_spatial_bin_x(checked_cast<uint32_t> (_spatial_bins_x));
    builder.add_spatial_bin_y(checked_cast<uint32_t> (_spatial_bins_y));
    builder.add_mode         (PSROIPoolingMode2Int32(_mode));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PSROIPoolingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parsePSROIPooling(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                          ArrayRef<mlir::Value> outputs,
                                                          const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAPSROIPooling supports only 2 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAPSROIPooling supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_PSROIPoolingParams();
    const auto output_dim     = getIntAttr(_ctx, params->output_dim());
    const auto spatialScale   = getFPAttr (_ctx, params->spatial_scale());
    const auto group_size     = getIntAttr(_ctx, params->group_size());
    const auto spatial_bins_x = getIntAttr(_ctx, params->spatial_bin_x());
    const auto spatial_bins_y = getIntAttr(_ctx, params->spatial_bin_y());

    IE::PSROIPoolingMode mode;
    switch (params->mode()) {
    case 0:
        mode = IE::PSROIPoolingMode::average;
        break;
    case 1:
        mode = IE::PSROIPoolingMode::bilinear;
        break;
    default:
        VPUX_THROW("Unknown PSROIPoolingMode. average and bilinear mode are supported only");
    }

    return builder.create<VPUIP::PSROIPoolingUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                  output_dim, spatialScale, group_size, spatial_bins_x, spatial_bins_y,
                                                  IE::PSROIPoolingModeAttr::get(_ctx, mode));
}
