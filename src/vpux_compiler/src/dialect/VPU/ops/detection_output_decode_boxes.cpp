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

mlir::LogicalResult VPU::DetectionOutputDecodeBoxesOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputDecodeBoxesOpAdaptor decodeBoxes(operands, attrs);
    if (mlir::failed(decodeBoxes.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = decodeBoxes.box_logits().getType().cast<NDTypeInterface>();
    inferredReturnTypes.push_back(inputType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputDecodeBoxesOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputDecodeBoxesOp is not supported by EMU");
}
