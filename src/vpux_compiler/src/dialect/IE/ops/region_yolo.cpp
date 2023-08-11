//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::RegionYoloOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RegionYoloOpAdaptor regionYolo(operands, attrs);
    if (mlir::failed(regionYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = regionYolo.input().getType().cast<mlir::ShapedType>();

    SmallVector<int64_t> outputShape;
    if (regionYolo.do_softmax()) {
        for (int64_t i = 0; i < regionYolo.axis(); i++) {
            outputShape.push_back(inType.getShape()[i]);
        }

        size_t flat_dim = 1;
        for (int64_t i = regionYolo.axis(); i < regionYolo.end_axis() + 1; i++) {
            flat_dim *= inType.getShape()[i];
        }
        outputShape.push_back(flat_dim);

        for (size_t i = regionYolo.end_axis() + 1; i < inType.getShape().size(); i++) {
            outputShape.push_back(inType.getShape()[i]);
        }
    } else {
        outputShape.push_back(inType.getShape()[0]);
        outputShape.push_back((regionYolo.classes() + regionYolo.coords() + 1) *
                              checked_cast<int64_t>(regionYolo.mask().size()));
        outputShape.push_back(inType.getShape()[2]);
        outputShape.push_back(inType.getShape()[3]);
    }

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());
    return mlir::success();
}
