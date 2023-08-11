//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::NormalizeL2Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NormalizeL2OpAdaptor normalizeL2(operands, attrs);
    if (mlir::failed(normalizeL2.verify(loc))) {
        return mlir::failure();
    }

    auto axes = IE::constInputToData(loc, normalizeL2.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto inType = normalizeL2.data().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::NormalizeL2Op::verify() {
    const auto inRank = data().getType().cast<mlir::ShapedType>().getRank();
    auto axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(axes()));

    for (auto& axis : axesVec) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(*this, "Axes values should be unique");
    }

    return mlir::success();
}
