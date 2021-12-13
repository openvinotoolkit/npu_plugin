//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ProposalOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ProposalOpAdaptor proposal(operands, attrs);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = proposal.class_probs().getType().cast<mlir::ShapedType>();

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{inType.getShape().front() * proposal.proposal_attrs().postNmsTopN().getInt(),
                                        5};
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::ProposalOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ProposalParamsBuilder builder(writer);

    IE::ProposalAttr attr = proposal_attrs();
    EMU::BlobWriter::String framework = writer.createString(attr.framework().getValue());

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

