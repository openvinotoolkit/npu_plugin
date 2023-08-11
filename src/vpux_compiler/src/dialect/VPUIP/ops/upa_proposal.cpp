//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::ProposalUPAOp::verify() {
    IE::ProposalAttr attr = proposal_attrs();
    if (attr.framework().getValue() != "" && attr.framework().getValue() != "tensorflow") {
        return errorAt(*this, "Unsupported framework attr {0}", attr.framework().getValue().str());
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ProposalUPAOp::serialize(VPUIP::BlobWriter& writer) {
    IE::ProposalAttr attr = proposal_attrs();
    VPUIP::BlobWriter::String framework = writer.createString(attr.framework().getValue());

    auto ratio_fb = writer.createVector(parseFPArrayAttr<float>(attr.ratio()));
    auto scale_fb = writer.createVector(parseFPArrayAttr<float>(attr.scale()));

    MVCNN::ProposalParamsBuilder builder(writer);
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
