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

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
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
    builder.add_input_height(checked_cast<uint32_t>(detectionOutputAttr.input_height().getUInt()));
    builder.add_input_width(checked_cast<uint32_t>(detectionOutputAttr.input_width().getUInt()));
    builder.add_objectness_score(
            static_cast<float>(detectionOutputAttr.objectness_score().getValue().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DetectionOutputParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseDetectionOutput(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                               ArrayRef<mlir::Value> outputs,
                                                               const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 5, "UPADetectionOutput supports only 5 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPADetectionOutput supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_DetectionOutputParams();
    const auto numClasses = getIntAttr(_ctx, params->num_classes());
    const auto backgroundLabelId = getIntAttr(_ctx, params->background_label_id());
    const auto topK = getIntAttr(_ctx, params->top_k());
    const auto varianceEncodedInTarget = mlir::BoolAttr::get(_ctx, params->variance_encoded_in_target());
    const auto keepTopK = getIntArrayAttr(_ctx, SmallVector<int32_t>{params->keep_top_k()});
    const auto codeType = mlir::StringAttr::get(_ctx, params->code_type()->str());
    const auto shareLocation = mlir::BoolAttr::get(_ctx, params->share_location());
    const auto nmsThreshold = getFPAttr(_ctx, params->nms_threshold());
    const auto confidenceThreshold = getFPAttr(_ctx, params->confidence_threshold());
    const auto clipAfterNms = mlir::BoolAttr::get(_ctx, params->clip_after_nms());
    const auto clipBeforeNms = mlir::BoolAttr::get(_ctx, params->clip_before_nms());
    const auto decreaseLabelId = mlir::BoolAttr::get(_ctx, params->decrease_label_id());
    const auto normalized = mlir::BoolAttr::get(_ctx, params->normalized());
    const auto inputHeight = getIntAttr(_ctx, params->input_height());
    const auto inputWidth = getIntAttr(_ctx, params->input_width());
    const auto objectnessScore = getFPAttr(_ctx, params->objectness_score());

    const auto detectionOutputAttr = IE::DetectionOutputAttr::get(
            numClasses, backgroundLabelId, topK, varianceEncodedInTarget, keepTopK, codeType, shareLocation,
            nmsThreshold, confidenceThreshold, clipAfterNms, clipBeforeNms, decreaseLabelId, normalized, inputHeight,
            inputWidth, objectnessScore, _ctx);
    return builder.create<VPUIP::DetectionOutputUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2],
                                                       inputs[3], inputs[4], outputs[0], detectionOutputAttr);
}
