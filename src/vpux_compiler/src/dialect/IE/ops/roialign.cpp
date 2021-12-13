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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ROIAlignOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ROIAlignOpAdaptor roiAlign(operands, attrs);
    if (mlir::failed(roiAlign.verify(loc))) {
        return mlir::failure();
    }

    const auto pooled_h = roiAlign.pooled_h().getInt();
    const auto pooled_w = roiAlign.pooled_w().getInt();
    const auto inTypeFeatureMap = roiAlign.input().getType().cast<mlir::ShapedType>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = roiAlign.coords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (pooled_h <= 0) {
        return errorAt(loc, "Pooled_h should be positive. Got {0}", pooled_h);
    }

    if (pooled_w <= 0) {
        return errorAt(loc, "Pooled_w should be positive. Got {0}", pooled_w);
    }

    SmallVector<int64_t> output_shape;
    output_shape.push_back(inShapeCoord[0]);
    output_shape.push_back(inShapeFeatureMap[1]);
    output_shape.push_back(pooled_h);
    output_shape.push_back(pooled_w);

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());
    return mlir::success();
}

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


//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::ROIAlignOp::serialize(EMU::BlobWriter& writer) {
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
