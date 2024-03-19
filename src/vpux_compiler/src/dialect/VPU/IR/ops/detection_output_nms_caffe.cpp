// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputNmsCaffeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputNmsCaffeOpAdaptor nmsCaffe(operands, attrs);
    if (mlir::failed(nmsCaffe.verify(loc))) {
        return mlir::failure();
    }

    const auto topKConfidenceType = nmsCaffe.getTopKConfidence().getType().cast<NDTypeInterface>();
    const auto boxesType = nmsCaffe.getBoxes().getType().cast<NDTypeInterface>();
    const auto sizesType = nmsCaffe.getSizes().getType().cast<NDTypeInterface>();

    inferredReturnTypes.push_back(topKConfidenceType);
    inferredReturnTypes.push_back(boxesType);
    inferredReturnTypes.push_back(sizesType);

    return mlir::success();
}
