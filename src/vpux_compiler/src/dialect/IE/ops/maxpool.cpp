//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/propagate_quantize_dequantize_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/validation_util.hpp>

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.pads_begin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.strides());
    const auto roundingType = maxPool.rounding_type();

    const auto inType = maxPool.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = maxPool.input().getType().cast<mlir::ShapedType>().getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()), true, roundingType == RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inType);

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::MaxPoolOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeDown(info);
}

void vpux::IE::MaxPoolOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeUp(info);
}
