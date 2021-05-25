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
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::DetectionOutputUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                              mlir::Value in_box_logits, mlir::Value in_class_preds,
                                              mlir::Value in_proposals, mlir::Value in_additional_preds,
                                              mlir::Value in_additional_proposals, mlir::Value output,
                                              vpux::IE::DetectionOutputAttr attr) {
    build(builder, state, in_box_logits, in_class_preds, in_proposals, in_additional_preds, in_additional_proposals,
          output, mlir::ValueRange{}, mlir::ValueRange{}, attr, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DetectionOutputUPAOp::serialize(VPUIP::BlobWriter& writer) {
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
    builder.add_nms_threshold(detectionOutputAttr.nms_threshold().getValue().convertToFloat());
    builder.add_background_label_id(checked_cast<int32_t>(detectionOutputAttr.background_label_id().getInt()));
    builder.add_top_k(checked_cast<int32_t>(detectionOutputAttr.top_k().getInt()));
    builder.add_variance_encoded_in_target(detectionOutputAttr.variance_encoded_in_target().getValue());
    builder.add_code_type(fb_code_type);
    builder.add_share_location(detectionOutputAttr.share_location().getValue());
    builder.add_confidence_threshold(detectionOutputAttr.confidence_threshold().getValue().convertToFloat());
    builder.add_clip_before_nms(detectionOutputAttr.clip_before_nms().getValue());
    builder.add_clip_after_nms(detectionOutputAttr.clip_after_nms().getValue());
    builder.add_decrease_label_id(detectionOutputAttr.decrease_label_id().getValue());
    builder.add_normalized(detectionOutputAttr.normalized().getValue());
    builder.add_input_height(checked_cast<uint32_t>(detectionOutputAttr.input_height().getUInt()));
    builder.add_input_width(checked_cast<uint32_t>(detectionOutputAttr.input_width().getUInt()));
    builder.add_objectness_score(detectionOutputAttr.objectness_score().getValue().convertToFloat());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DetectionOutputParams});
}
