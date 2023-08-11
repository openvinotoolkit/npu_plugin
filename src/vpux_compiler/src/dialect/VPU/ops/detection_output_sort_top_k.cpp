//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputSortTopKOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputSortTopKOpAdaptor sortTopK(operands, attrs);
    if (mlir::failed(sortTopK.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = sortTopK.class_predictions().getType().cast<NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    const auto batchSize = inputShape[Dim(0)];
    const auto numClasses = inputShape[Dim(1)];
    const auto numPriors = inputShape[Dim(2)];

    const auto topK = sortTopK.top_k();

    const auto outTopKConfidenceShape = SmallVector<int64_t>{batchSize, numClasses, topK};
    const auto outIndicesShape = SmallVector<int64_t>{batchSize, numClasses, numPriors};
    const auto outSizesShape = SmallVector<int64_t>{batchSize, numClasses};

    const auto outTopKConfidenceType = inputType.changeShape(Shape(outTopKConfidenceShape));

    const auto outIndicesElemType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto outIndicesType = mlir::RankedTensorType::get(outIndicesShape, outIndicesElemType);
    const auto outSizesType = mlir::RankedTensorType::get(outSizesShape, outIndicesElemType);

    inferredReturnTypes.push_back(outTopKConfidenceType);
    inferredReturnTypes.push_back(outIndicesType);
    inferredReturnTypes.push_back(outSizesType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask VPU::DetectionOutputSortTopKOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("VPU::DetectionOutputSortTopKOp is not supported by EMU");
}
