//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::PermuteQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PermuteQuantizeOpAdaptor permute_quantize(operands, attrs);
    if (mlir::failed(permute_quantize.verify(loc))) {
        return mlir::failure();
    }

    mlir::Value input = permute_quantize.input();
    mlir::AffineMap memPerm = permute_quantize.mem_perm();
    mlir::AffineMap dstOrder = permute_quantize.dst_order();
    const auto dstElemType = permute_quantize.dstElemType();

    const auto padBegin = parseIntArrayAttr<int64_t>(permute_quantize.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(permute_quantize.pads_end());

    const auto inType = permute_quantize.input().getType().cast<vpux::NDTypeInterface>();

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dstOrder);

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    const auto inShapeExpanded = newType.getShape();

    const auto inMemShape = inOrder.toMemoryOrder(inShapeExpanded);
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    const auto outDesc = IE::getTensorAttr(dstOrder, nullptr);

    inferredReturnShapes.emplace_back(outShape.raw(), dstElemType, outDesc);

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::PermuteQuantizeOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType() && mem_perm().isIdentity()) {
        return input();
    }

    return nullptr;
}
