//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::VPU::AvgPoolOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AvgPoolOpAdaptor avgPool(operands, attrs);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.getStrides());
    const auto roundingType = avgPool.getRoundingType();

    const auto inType = avgPool.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            EmptyNode::instance(), ov::Shape(inShape.begin(), inShape.end()),
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ov::Shape(windowShape.begin(), windowShape.end()), ov::Strides(windowStrides.begin(), windowStrides.end()),
            true, /* It is only used during assertion. True will make it pass */
            roundingType == vpux::IE::RoundingType::CEIL);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));

    const auto outType = inType.changeShape(Shape(shapeI64));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::AvgPoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(getInput());
    const auto pads = PadInfo(getPadsBegin(), getPadsEnd());
    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, getKernelSize(), getStrides(), pads);
    return inputTiling;
}

void vpux::VPU::AvgPoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    IE::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::AvgPoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
