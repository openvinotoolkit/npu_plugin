//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {
// This method converts value from ROIPoolingMethod view to corresponds t_ROIPooling_method view from runtime
uint32_t ROIPoolingMethod2Int32(IE::ROIPoolingMethod method) {
    uint32_t out_code = 0;
    switch (method) {
    case IE::ROIPoolingMethod::max:
        out_code = 0;
        break;
    case IE::ROIPoolingMethod::bilinear:
        out_code = 1;
        break;
    default:
        VPUX_THROW("Unknown ROIPoolingMethod. max and bilinear methods are supported only");
    }
    return out_code;
}
}  // namespace
mlir::LogicalResult vpux::VPUIP::verifyOp(ROIPoolingUPAOp op) {
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

    const auto output_size = parseIntArrayAttr(op.output_size());
    if (output_size.size() != 2) {
        return errorAt(op, "Dimension of pooled size is expected to be equal to 2. Got {0}", output_size.size());
    }

    if (output_size[0] <= 0 || output_size[1] <= 0) {
        return errorAt(op, "Pooled size attributes pooled_h and pooled_w should be positive.");
    }

    return mlir::success();
}

void vpux::VPUIP::ROIPoolingUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                         mlir::Value input, mlir::Value coords, mlir::Value output,
                                         mlir::ArrayAttr output_size, mlir::FloatAttr spatial_scale,
                                         IE::ROIPoolingMethodAttr method) {
    build(odsBuilder, odsState, input, coords, output, mlir::ValueRange{}, mlir::ValueRange{}, output_size,
          spatial_scale, method, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ROIPoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    float spatial_scale_val = spatial_scale().convertToFloat();
    uint32_t num_rois = checked_cast<uint32_t>(coords().getType().cast<mlir::ShapedType>().getShape()[0]);
    const auto output_size = parseIntArrayAttr(output_sizeAttr());

    MVCNN::ROIPoolingParamsBuilder builder(writer);
    builder.add_spatial_scale(spatial_scale_val);
    builder.add_roi_pooling_method(ROIPoolingMethod2Int32(method()));
    builder.add_num_rois(num_rois);
    builder.add_pooled_h(checked_cast<uint32_t>(output_size[0]));
    builder.add_pooled_w(checked_cast<uint32_t>(output_size[1]));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ROIPoolingParams});
}
