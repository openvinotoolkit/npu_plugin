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

mlir::LogicalResult vpux::VPU::PSROIPoolingOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::PSROIPoolingOpAdaptor psroiPooling(operands, attrs);
    if (mlir::failed(psroiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputDim = psroiPooling.output_dim().getInt();
    const auto groupSize = psroiPooling.group_size().getInt();
    const auto inTypeFeatureMap = psroiPooling.input().getType().cast<vpux::NDTypeInterface>();
    const auto inTypeCoord = psroiPooling.coords().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeCoord = inTypeCoord.getShape();

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

EMU::BlobWriter::SpecificTask vpux::VPU::PSROIPoolingOp::serialize(EMU::BlobWriter& writer) {
    const auto spatialBinsX = spatial_bins_x().hasValue() ? spatial_bins_x().getValue() : 1;
    const auto spatialBinsY = spatial_bins_y().hasValue() ? spatial_bins_y().getValue() : 1;
    const auto psROIPoolingMode = mode().hasValue() ? mode().getValue() : IE::PSROIPoolingMode::AVERAGE;

    MVCNN::PSROIPoolingParamsBuilder builder(writer);

    builder.add_output_dim(checked_cast<uint32_t>(output_dim()));
    builder.add_spatial_scale(checked_cast<float>(spatial_scaleAttr().getValueAsDouble()));
    builder.add_group_size(checked_cast<uint32_t>(group_size()));
    builder.add_pooled_w(checked_cast<uint32_t>(group_size()));
    builder.add_pooled_h(checked_cast<uint32_t>(group_size()));
    builder.add_spatial_bin_x(checked_cast<uint32_t>(spatialBinsX));
    builder.add_spatial_bin_y(checked_cast<uint32_t>(spatialBinsY));
    builder.add_mode(VPUIP::convertVPUXPSROIPoolingModeToMVNCNN(psROIPoolingMode));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PSROIPoolingParams});
}
