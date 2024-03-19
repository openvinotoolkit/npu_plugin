//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.getStrides());
    const auto roundingType = maxPool.getRoundingType();

    const auto inType = maxPool.getInput().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = maxPool.getInput().getType().cast<mlir::ShapedType>().getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            EmptyNode::instance(), ov::Shape(inShape.begin(), inShape.end()),
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ov::Shape(windowShape.begin(), windowShape.end()), ov::Strides(windowStrides.begin(), windowStrides.end()),
            true, roundingType == RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}
