//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ROIAlignOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ROIAlignOpAdaptor roiAlign(operands, attrs);
    if (mlir::failed(roiAlign.verify(loc))) {
        return mlir::failure();
    }

    const auto pooledH = roiAlign.pooled_h();
    const auto pooledW = roiAlign.pooled_w();
    const auto inTypeFeatureMap = roiAlign.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = roiAlign.coords().getType().cast<vpux::NDTypeInterface>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (pooledH <= 0) {
        return errorAt(loc, "pooledH should be positive. Got {0}", pooledH);
    }

    if (pooledW <= 0) {
        return errorAt(loc, "pooledW should be positive. Got {0}", pooledW);
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inShapeCoord.raw()[0]);
    outputShape.push_back(inShapeFeatureMap.raw()[1]);
    outputShape.push_back(pooledH);
    outputShape.push_back(pooledW);

    const auto outType = inTypeFeatureMap.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ROIAlignOp::serialize(EMU::BlobWriter& writer) {
    const float spatial_scale_val = static_cast<float>(spatial_scale().convertToDouble());

    MVCNN::ROIAlignParamsBuilder builder(writer);
    builder.add_spatial_scale(spatial_scale_val);
    builder.add_method(VPUIP::convertVPUXROIAlignMethod2MVCNN(poolingMode()));
    builder.add_sampling_ratio(checked_cast<uint32_t>(sampling_ratio()));
    builder.add_pooled_h(checked_cast<uint32_t>(pooled_h()));
    builder.add_pooled_w(checked_cast<uint32_t>(pooled_w()));
    builder.add_roi_align_step(MVCNN::ROIAlignStep_roi_align);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ROIAlignParams});
}
