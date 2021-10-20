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
#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

namespace {
MVCNN::ROIAlignMethod ROIAlignMethod2MVCNN(IE::ROIAlignMethod method) {
    MVCNN::ROIAlignMethod mvcnn_method;
    switch (method) {
        case IE::ROIAlignMethod::avg:
            mvcnn_method = MVCNN::ROIAlignMethod_roi_align_avg;
            break;
        case IE::ROIAlignMethod::max:
            mvcnn_method = MVCNN::ROIAlignMethod_roi_align_max;
            break;
        default:
            VPUX_THROW("Unknown ROIAlignMethod. avg and max methods are supported only");
    }
    return mvcnn_method;
}
}

void vpux::VPUIP::ROIAlignUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                       mlir::Value input, mlir::Value coords, mlir::Value roisIdx, mlir::Value output,
                                       mlir::IntegerAttr pooled_h, mlir::IntegerAttr pooled_w, mlir::IntegerAttr sampling_ratio,
                                       mlir::FloatAttr spatial_scale, IE::ROIAlignMethodAttr poolingMode) {
    build(odsBuilder, odsState, input, coords, roisIdx, output, mlir::ValueRange{}, mlir::ValueRange{}, pooled_h, pooled_w,
          sampling_ratio, spatial_scale, poolingMode, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ROIAlignUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float spatial_scale_val = static_cast<float>(spatial_scale().convertToDouble());

    MVCNN::ROIAlignParamsBuilder builder(writer);
    builder.add_spatial_scale(spatial_scale_val);
    builder.add_method(ROIAlignMethod2MVCNN(poolingMode()));
    builder.add_sampling_ratio(sampling_ratio());
    builder.add_pooled_h(pooled_h());
    builder.add_pooled_w(pooled_w());
    builder.add_roi_align_step(MVCNN::ROIAlignStep_roi_align);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ROIAlignParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseROIAlign(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                          ArrayRef<mlir::Value> outputs,
                                                          const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 3, "UPAROIAlign supports only 3 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAROIAlign supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ROIAlignParams();
    const auto pooled_h = getIntAttr(_ctx, params->pooled_h());
    const auto pooled_w = getIntAttr(_ctx, params->pooled_w());
    const auto sampling_ratio = getIntAttr(_ctx, params->sampling_ratio());
    const auto spatial_scale = getFPAttr(_ctx, params->spatial_scale());
    IE::ROIAlignMethod method;
    switch (params->method()) {
    case 0:
        method = IE::ROIAlignMethod::avg;
        break;
    case 1:
        method = IE::ROIAlignMethod::max;
        break;
    default:
       VPUX_THROW("Unknown ROIAlignMethod. avg and max methods are supported only");
    }

    return builder.create<VPUIP::ROIAlignUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2], outputs[0],
                                                  pooled_h, pooled_w, sampling_ratio, spatial_scale,
                                                  IE::ROIAlignMethodAttr::get(_ctx, method));
}