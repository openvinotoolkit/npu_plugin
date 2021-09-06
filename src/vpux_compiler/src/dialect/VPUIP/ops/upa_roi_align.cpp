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
    MVCNN::ROIAlignMethod out_code;
    switch (method) {
        case IE::ROIAlignMethod::avg:
            out_code = MVCNN::ROIAlignMethod_roi_align_avg;
            break;
        case IE::ROIAlignMethod::max:
            out_code = MVCNN::ROIAlignMethod_roi_align_max;
            break;
        default:
            VPUX_THROW("Unknown ROIAlignMethod. avg and max methods are supported only");
    }
    return out_code;
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

