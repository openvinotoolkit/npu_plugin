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

#include "vpux/compiler/dialect/EMU/ops.hpp"

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

EMU::BlobWriter::SpecificTask vpux::EMU::ROIAlignUPAOp::serialize(EMU::BlobWriter& writer) {
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
