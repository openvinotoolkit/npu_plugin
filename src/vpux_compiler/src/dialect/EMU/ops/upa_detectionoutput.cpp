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

#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

EMU::BlobWriter::SpecificTask vpux::EMU::DetectionOutputUPAOp::serialize(EMU::BlobWriter& writer) {
    const auto detectionOutputAttr = attr();
    const auto code_type = detectionOutputAttr.code_type().getValue().str();

    std::string code_type_upa{"CORNER"};
    if (code_type == "caffe.PriorBoxParameter.CORNER_SIZE")
        code_type_upa = "CORNER_SIZE";
    else if (code_type == "caffe.PriorBoxParameter.CENTER_SIZE")
        code_type_upa = "CENTER_SIZE";

    const auto fb_code_type = writer.createString(code_type_upa);

    MVCNN::DetectionOutputParamsBuilder builder(writer);
    builder.add_num_classes(checked_cast<int32_t>(detectionOutputAttr.num_classes().getInt()));
    builder.add_keep_top_k(
            checked_cast<int32_t>(detectionOutputAttr.keep_top_k()[0].cast<mlir::IntegerAttr>().getInt()));
    builder.add_nms_threshold(static_cast<float>(detectionOutputAttr.nms_threshold().getValue().convertToDouble()));
    builder.add_background_label_id(checked_cast<int32_t>(detectionOutputAttr.background_label_id().getInt()));
    builder.add_top_k(checked_cast<int32_t>(detectionOutputAttr.top_k().getInt()));
    builder.add_variance_encoded_in_target(detectionOutputAttr.variance_encoded_in_target().getValue());
    builder.add_code_type(fb_code_type);
    builder.add_share_location(detectionOutputAttr.share_location().getValue());
    builder.add_confidence_threshold(
            static_cast<float>(detectionOutputAttr.confidence_threshold().getValue().convertToDouble()));
    builder.add_clip_before_nms(detectionOutputAttr.clip_before_nms().getValue());
    builder.add_clip_after_nms(detectionOutputAttr.clip_after_nms().getValue());
    builder.add_decrease_label_id(detectionOutputAttr.decrease_label_id().getValue());
    builder.add_normalized(detectionOutputAttr.normalized().getValue());
    builder.add_input_height(checked_cast<uint32_t>(detectionOutputAttr.input_height().getValue().getSExtValue()));
    builder.add_input_width(checked_cast<uint32_t>(detectionOutputAttr.input_width().getValue().getSExtValue()));
    builder.add_objectness_score(
            static_cast<float>(detectionOutputAttr.objectness_score().getValue().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DetectionOutputParams});
}
