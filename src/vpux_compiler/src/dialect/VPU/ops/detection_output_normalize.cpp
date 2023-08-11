//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::VPU::DetectionOutputNormalizeOp::verify() {
    const auto inType = prior_boxes().getType().cast<NDTypeInterface>();
    const auto inputShape = inType.getShape();
    if (inputShape.size() != 3 && inputShape[Dim(1)] != 1) {
        return errorAt(*this, "Input shape must be 3D with height == 1, got {0}", inputShape);
    }

    return mlir::success();
}

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputNormalizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputNormalizeOpAdaptor normalize(operands, attrs);
    if (mlir::failed(normalize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = normalize.prior_boxes().getType().cast<NDTypeInterface>();
    inferredReturnTypes.push_back(inputType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputNormalizeOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputNormalizeOp is not supported by EMU");
}
