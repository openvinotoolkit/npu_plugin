//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DeformablePSROIPoolingOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DeformablePSROIPoolingOpAdaptor deformablepsroiPooling(operands, attrs);
    if (mlir::failed(deformablepsroiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputDim = deformablepsroiPooling.output_dim();
    const auto groupSize =
            deformablepsroiPooling.group_size().has_value() ? deformablepsroiPooling.group_size().value() : 1;
    const auto inTypeFeatureMap = deformablepsroiPooling.input_score_maps().getType().cast<vpux::NDTypeInterface>();
    const auto inType2 = deformablepsroiPooling.input_rois().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeCoord = inType2.getShape();

    if (outputDim <= 0) {
        return errorAt(loc, "Pooled size attribute outputDim should be positive.");
    }

    if (groupSize <= 0) {
        return errorAt(loc, "Group size attribute groupSize should be positive.");
    }

    SmallVector<int64_t> outputShape({inShapeCoord.raw()[0],  // num_rois
                                      outputDim,              // output channel number
                                      groupSize,              // pooled_w
                                      groupSize});            // pooled_h

    const auto outType = inTypeFeatureMap.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DeformablePSROIPoolingOp::serialize(EMU::BlobWriter& writer) {
    const auto spatialBinsX = spatial_bins_x().has_value() ? spatial_bins_x().value() : 1;
    const auto spatialBinsY = spatial_bins_y().has_value() ? spatial_bins_y().value() : 1;
    const auto groupSize = group_size().has_value() ? group_size().value() : 1;
    const auto partSize = part_size().has_value() ? part_size().value() : 0;
    const auto transStd = trans_std().has_value();
    const auto deformablepsROIPoolingMode =
            mode().has_value() ? mode().value() : IE::DeformablePSROIPoolingMode::BILINEAR_DEFORMABLE;

    MVCNN::DeformablePSROIPoolingParamsBuilder builder(writer);

    builder.add_output_dim(checked_cast<uint32_t>(output_dim()));
    builder.add_spatial_scale(checked_cast<float>(spatial_scaleAttr().getValueAsDouble()));
    builder.add_group_size(checked_cast<uint32_t>(groupSize));
    builder.add_spatial_bins_x(checked_cast<uint32_t>(spatialBinsX));
    builder.add_spatial_bins_y(checked_cast<uint32_t>(spatialBinsY));
    builder.add_trans_std(checked_cast<float>(transStd));
    builder.add_part_size(checked_cast<uint32_t>(partSize));
    builder.add_mode(VPUIP::convertVPUXDeformablePSROIPoolingModeToMVNCNN(deformablepsROIPoolingMode));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this,
                                     {paramsOff.Union(), MVCNN::SoftwareLayerParams_DeformablePSROIPoolingParams});
}
