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

mlir::LogicalResult VPU::DetectionOutputCollectResultsOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputCollectResultsOpAdaptor collectResults(operands, attrs);
    if (mlir::failed(collectResults.verify(loc))) {
        return mlir::failure();
    }

    const auto keepTopK = collectResults.keep_top_k();

    const auto confidenceType = collectResults.confidence().getType().cast<NDTypeInterface>();
    const auto detectionSize = 7;
    const auto outputShape = SmallVector<int64_t>{1, 1, keepTopK, detectionSize};
    const auto outputType = confidenceType.changeShape(Shape(outputShape));

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputCollectResultsOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputCollectResultsOp is not supported by EMU");
}
