//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherElementsOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GatherElementsOpAdaptor gatherElements(operands, attrs);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = gatherElements.indices().getType().cast<mlir::ShapedType>();
    const auto inInputType = gatherElements.input().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inIndicesType.getShape(), inInputType.getElementType());
    return mlir::success();
}
