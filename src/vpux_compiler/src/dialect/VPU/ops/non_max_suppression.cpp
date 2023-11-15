//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NonMaxSuppressionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NonMaxSuppressionOpAdaptor nms(operands, attrs);
    if (mlir::failed(nms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = nms.in_box_scores().getType().cast<vpux::NDTypeInterface>();
    const auto sInt32Type = inType.changeElemType(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed));

    int64_t maxOutputBoxesPerClass = nms.max_output_boxes_per_class_valueAttr().getValue().getSExtValue();
    const auto inShape = inType.getShape().raw();  // nbatch*nclasses*nboxes
    const auto numBatches = inShape[0];
    const auto numClasses = inShape[1];
    const auto numBoxes = inShape[2];
    const auto minBoxes = std::min(numBoxes, maxOutputBoxesPerClass);
    const SmallVector<int64_t> outShape{minBoxes * numBatches * numClasses, 3};
    const SmallVector<int64_t> validOutputsShape{1};

    const auto outFloatType = inType.changeShape(Shape(outShape));
    const auto outIntType = sInt32Type.changeShape(Shape(outShape));
    const auto validOutputsType = sInt32Type.changeShape(Shape(validOutputsShape));
    inferredReturnTypes.push_back(outIntType);
    inferredReturnTypes.push_back(outFloatType);
    inferredReturnTypes.push_back(validOutputsType);
    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NonMaxSuppressionOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::NMSParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NMSParams});
}
