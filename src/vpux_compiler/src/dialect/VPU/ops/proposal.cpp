//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ProposalOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ProposalOpAdaptor proposal(operands, attrs);
    if (mlir::failed(proposal.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = proposal.class_probs().getType().cast<vpux::NDTypeInterface>();

    // out shape must be [batch_size * post_nms_topn, 5]
    const SmallVector<int64_t> outShape{inType.getShape().front() * proposal.proposal_attrs().postNmsTopN().getInt(),
                                        5};
    const SmallVector<int64_t> probsShape{inType.getShape().front() * proposal.proposal_attrs().postNmsTopN().getInt()};

    const auto outType = inType.changeShape(Shape(outShape));
    const auto probsType = inType.changeShape(Shape(probsShape));
    inferredReturnTypes.push_back(outType);
    inferredReturnTypes.push_back(probsType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ProposalOp::serialize(EMU::BlobWriter& writer) {
    IE::ProposalAttr attr = proposal_attrs();
    EMU::BlobWriter::String framework = writer.createString(attr.framework().getValue());

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
