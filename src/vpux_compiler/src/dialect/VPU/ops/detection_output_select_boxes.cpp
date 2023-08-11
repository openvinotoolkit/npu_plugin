//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputSelectBoxesOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputSelectBoxesOpAdaptor selectBoxes(operands, attrs);
    if (mlir::failed(selectBoxes.verify(loc))) {
        return mlir::failure();
    }

    const auto indicesType = selectBoxes.indices().getType().cast<NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();

    const auto numPriors = indicesShape[Dim(2)];
    const auto topK = selectBoxes.top_k();

    const auto numOutBoxes = std::min(topK, numPriors);

    const auto boxesType = selectBoxes.decoded_boxes().getType().cast<NDTypeInterface>();
    const auto outputShape = SmallVector<int64_t>{indicesShape[Dim(0)], indicesShape[Dim(1)], numOutBoxes * 4};
    const auto outputType = boxesType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputSelectBoxesOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputSelectBoxesOp is not supported by EMU");
}
