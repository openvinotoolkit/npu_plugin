//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherSliceOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherSliceOpAdaptor gatherSlice(operands, attrs);
    if (mlir::failed(gatherSlice.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gatherSlice.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();
    const auto indicesType = gatherSlice.indices().getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape().raw();

    const auto axisVal = gatherSlice.axis_valueAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();

    SmallVector<int64_t> outShape;

    // calculate output shape
    int64_t batchDims = gatherSlice.batch_dims();
    int64_t outRank = inputShape.size() + indicesShape.size() - 1 - batchDims;
    int64_t indicesRank = indicesShape.size();
    int64_t i = 0;

    for (; i < batchDims; i++) {
        outShape.push_back(inputShape[i] & indicesShape[i]);
    }
    for (; i < axisVal; i++) {
        outShape.push_back(inputShape[i]);
    }
    for (; i < axisVal + indicesRank - batchDims; i++) {
        outShape.push_back(indicesShape[batchDims - axisVal + i]);
    }
    for (; i < outRank; i++) {
        outShape.push_back(inputShape[batchDims + 1 - indicesRank + i]);
    }

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    const auto flagElementType = getBool8Type(ctx);
    const auto flagType = indicesType.changeElemType(flagElementType);
    inferredReturnTypes.push_back(flagType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::GatherSliceOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("GatherSliceOp implemented just on 37xx.");
}
