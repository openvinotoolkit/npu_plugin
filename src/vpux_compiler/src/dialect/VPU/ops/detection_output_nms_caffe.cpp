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

mlir::LogicalResult VPU::DetectionOutputNmsCaffeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputNmsCaffeOpAdaptor nmsCaffe(operands, attrs);
    if (mlir::failed(nmsCaffe.verify(loc))) {
        return mlir::failure();
    }

    const auto topKConfidenceType = nmsCaffe.top_k_confidence().getType().cast<NDTypeInterface>();
    const auto boxesType = nmsCaffe.boxes().getType().cast<NDTypeInterface>();
    const auto sizesType = nmsCaffe.sizes().getType().cast<NDTypeInterface>();

    inferredReturnTypes.push_back(topKConfidenceType);
    inferredReturnTypes.push_back(boxesType);
    inferredReturnTypes.push_back(sizesType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputNmsCaffeOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputNmsCaffeOp is not supported by EMU");
}
