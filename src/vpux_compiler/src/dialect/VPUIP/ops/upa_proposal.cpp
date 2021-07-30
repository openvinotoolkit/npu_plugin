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
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(ProposalUPAOp op) {
    IE::ProposalAttr attr = op.proposal_attrs();
    if (attr.framework().getValue() != "" && attr.framework().getValue() != "tensorflow") {
        return errorAt(op, "Unsupported framework attr {0}", attr.framework().getValue().str());
    }

    return mlir::success();
}

void vpux::VPUIP::ProposalUPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                       mlir::Value class_probs, mlir::Value bbox_deltas, mlir::Value image_shape,
                                       mlir::Value output, IE::ProposalAttr proposal_attrs) {
    build(odsBuilder, odsState, class_probs, bbox_deltas, image_shape, output, mlir::ValueRange{}, mlir::ValueRange{},
          proposal_attrs, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ProposalUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ProposalParamsBuilder builder(writer);

    IE::ProposalAttr attr = proposal_attrs();
    VPUIP::BlobWriter::String framework = writer.createString(attr.framework().getValue());

    auto ratio_fb = writer.createVector(parseFPArrayAttr<float>(attr.ratio()));
    auto scale_fb = writer.createVector(parseFPArrayAttr<float>(attr.scale()));

    builder.add_ratio(ratio_fb);
    builder.add_scale(scale_fb);
    builder.add_min_size(checked_cast<uint32_t>(attr.minSize().getValue().getSExtValue()));
    builder.add_base_size(checked_cast<uint32_t>(attr.baseSize().getValue().getSExtValue()));
    builder.add_framework(framework);
    builder.add_normalize(attr.normalize().getValue());
    builder.add_nms_thresh(static_cast<float>(attr.nmsThresh().getValueAsDouble()));
    builder.add_feat_stride(checked_cast<uint32_t>(attr.featStride().getValue().getSExtValue()));
    builder.add_pre_nms_topn(checked_cast<uint32_t>(attr.preNmsTopN().getValue().getSExtValue()));
    builder.add_post_nms_topn(checked_cast<uint32_t>(attr.postNmsTopN().getValue().getSExtValue()));
    builder.add_box_size_scale(static_cast<float>(attr.boxSizeScale().getValueAsDouble()));
    builder.add_clip_after_nms(attr.clipAfterNms().getValue());
    builder.add_for_deformable(false);  // ngraph doesn't have this parameter
    builder.add_pre_nms_thresh(0.0);    // ngraph doesn't have this parameter
    builder.add_clip_before_nms(attr.clipBeforeNms().getValue());
    builder.add_box_coordinate_scale(static_cast<float>(attr.boxCoordinateScale().getValueAsDouble()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ProposalParams});
}
