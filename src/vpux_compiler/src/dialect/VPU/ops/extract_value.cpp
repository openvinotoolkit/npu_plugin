//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExtractValueOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExtractValueOpAdaptor extractValue(operands, attrs);
    if (mlir::failed(extractValue.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = extractValue.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const int64_t inRank = inputShape.size();

    SmallVector<int64_t> outShape;
    for (int64_t i = 1; i < inRank; i++) {
        outShape.push_back(inputShape[i]);
    }

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ExtractValueOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("GatherSliceOp implemented just on 37xx.");
}
