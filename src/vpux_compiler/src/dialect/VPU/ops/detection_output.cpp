//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DetectionOutputOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputOpAdaptor detectionOutput(operands, attrs);
    if (mlir::failed(detectionOutput.verify(loc))) {
        return mlir::failure();
    }

    const auto boxLogitsType = detectionOutput.in_box_logits().getType().cast<vpux::NDTypeInterface>();

    auto origN{0}, origC{1};
    const auto numImages = boxLogitsType.getShape().raw()[origN];
    const auto numLocClasses =
            detectionOutput.attr().share_location().getValue() ? 1 : detectionOutput.attr().num_classes().getInt();

    if (numLocClasses <= 0) {
        return errorAt(loc, "Number of classes should be a natural number");
    }

    if (boxLogitsType.getShape().raw()[origC] % (numLocClasses * 4) != 0) {
        return errorAt(loc, "C dimension should be divisible by numLocClasses * 4");
    }

    const auto numPriorBoxes = boxLogitsType.getShape().raw()[origC] / (numLocClasses * 4);
    const auto keepTopK = detectionOutput.attr().keep_top_k()[0].cast<mlir::IntegerAttr>().getInt();
    const auto topK = detectionOutput.attr().top_k().getInt();
    const auto numClasses = detectionOutput.attr().num_classes().getInt();

    SmallVector<int64_t> outputShape{1, 1};
    if (keepTopK > 0) {
        outputShape.push_back(numImages * keepTopK);
    } else if (topK > 0) {
        outputShape.push_back(numImages * topK * numClasses);
    } else {
        outputShape.push_back(numImages * numPriorBoxes * numClasses);
    }
    outputShape.push_back(7);

    const auto outType = boxLogitsType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DetectionOutputOp::serialize(EMU::BlobWriter& writer) {
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
