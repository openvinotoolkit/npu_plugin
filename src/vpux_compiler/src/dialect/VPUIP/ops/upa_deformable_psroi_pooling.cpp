//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

IE::DeformablePSROIPoolingMode convertDeformablePSROIPoolingModeToIE(MVCNN::DeformablePSROIPoolingMode mode) {
    switch (mode) {
    case MVCNN::DeformablePSROIPoolingMode::DeformablePSROIPoolingMode_AVERAGE:
        return IE::DeformablePSROIPoolingMode::AVERAGE;
    case MVCNN::DeformablePSROIPoolingMode::DeformablePSROIPoolingMode_BILINEAR_DEFORMABLE:
        return IE::DeformablePSROIPoolingMode::BILINEAR_DEFORMABLE;
    default:
        VPUX_THROW("Unknown DeformablePSROIPoolingMode. Got {0} mode", mode);
    }
}

}  // namespace

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::DeformablePSROIPoolingUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto spatialBinsX = spatial_bins_x().hasValue() ? spatial_bins_x().getValue() : 1;
    const auto spatialBinsY = spatial_bins_y().hasValue() ? spatial_bins_y().getValue() : 1;
    const auto groupSize = group_size().hasValue() ? group_size().getValue() : 1;
    const auto partSize = part_size().hasValue() ? part_size().getValue() : 0;
    const auto deformablepsROIPoolingMode =
            mode().hasValue() ? mode().getValue() : IE::DeformablePSROIPoolingMode::BILINEAR_DEFORMABLE;

    MVCNN::DeformablePSROIPoolingParamsBuilder builder(writer);

    builder.add_output_dim(checked_cast<uint32_t>(output_dim()));
    builder.add_spatial_scale(checked_cast<float>(spatial_scaleAttr().getValueAsDouble()));
    builder.add_group_size(checked_cast<uint32_t>(groupSize));
    builder.add_spatial_bins_x(checked_cast<uint32_t>(spatialBinsX));
    builder.add_spatial_bins_y(checked_cast<uint32_t>(spatialBinsY));
    builder.add_trans_std(checked_cast<float>(trans_stdAttr().getValueAsDouble()));
    builder.add_part_size(checked_cast<uint32_t>(partSize));
    builder.add_mode(convertVPUXDeformablePSROIPoolingModeToMVNCNN(deformablepsROIPoolingMode));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this,
                                     {paramsOff.Union(), MVCNN::SoftwareLayerParams_DeformablePSROIPoolingParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseDeformablePSROIPooling(mlir::OpBuilder& builder,
                                                                      ArrayRef<mlir::Value> inputs,
                                                                      ArrayRef<mlir::Value> outputs,
                                                                      const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2 || inputs.size() == 3,
                      "UPADeformablePSROIPooling supports only 2 or 3 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPADeformablePSROIPooling supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_DeformablePSROIPoolingParams();
    const auto outputDim = getIntAttr(_ctx, params->output_dim());
    const auto spatialScale = getFPAttr(_ctx, params->spatial_scale());
    const auto groupSize = getIntAttr(_ctx, params->group_size());
    const auto spatialBinsX = getIntAttr(_ctx, params->spatial_bins_x());
    const auto spatialBinsY = getIntAttr(_ctx, params->spatial_bins_y());
    const auto transStd = getFPAttr(_ctx, params->trans_std());
    const auto partSize = getIntAttr(_ctx, params->part_size());
    IE::DeformablePSROIPoolingMode mode = convertDeformablePSROIPoolingModeToIE(params->mode());

    return builder.create<VPUIP::DeformablePSROIPoolingUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs.size() == 3 ? inputs[2] : nullptr, outputs[0],
            outputDim, spatialScale, groupSize, spatialBinsX, spatialBinsY, transStd, partSize,
            IE::DeformablePSROIPoolingModeAttr::get(_ctx, mode));
}
