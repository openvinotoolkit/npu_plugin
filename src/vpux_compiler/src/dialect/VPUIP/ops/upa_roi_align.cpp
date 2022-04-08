//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"

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

}  // namespace

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ROIAlignUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const float spatial_scale_val = static_cast<float>(spatial_scale().convertToDouble());

    MVCNN::ROIAlignParamsBuilder builder(writer);
    builder.add_spatial_scale(spatial_scale_val);
    builder.add_method(ROIAlignMethod2MVCNN(poolingMode()));
    builder.add_sampling_ratio(checked_cast<uint32_t>(sampling_ratio()));
    builder.add_pooled_h(checked_cast<uint32_t>(pooled_h()));
    builder.add_pooled_w(checked_cast<uint32_t>(pooled_w()));
    builder.add_roi_align_step(MVCNN::ROIAlignStep_roi_align);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ROIAlignParams});
}

namespace {

IE::ROIAlignMethod softLayerParam2IEMethod(size_t method) {
    IE::ROIAlignMethod ieMethod;
    switch (method) {
    case 0:
        ieMethod = IE::ROIAlignMethod::avg;
        break;
    case 1:
        ieMethod = IE::ROIAlignMethod::max;
        break;
    default:
        VPUX_THROW("Unknown ROIAlignMethod. avg and max methods are supported only");
    }

    return ieMethod;
}

}  // namespace

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
    IE::ROIAlignMethod method = softLayerParam2IEMethod(params->method());

    return builder.create<VPUIP::ROIAlignUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2],
                                                outputs[0], pooled_h, pooled_w, sampling_ratio, spatial_scale,
                                                IE::ROIAlignMethodAttr::get(_ctx, method));
}
