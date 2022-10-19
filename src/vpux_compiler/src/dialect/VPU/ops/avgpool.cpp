//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::VPU::AvgPoolOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                           mlir::Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::AvgPoolOpAdaptor avgPool(operands, attrs);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.pads_begin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.kernel_size());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.strides());
    const auto roundingType = avgPool.rounding_type().getValue();

    const auto inType = avgPool.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape();

    const auto outputShape = ngraph::infer_batched_pooling_forward(
            nullptr, ngraph::Shape(inShape.begin(), inShape.end()),
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()),
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
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::AvgPoolOp::serialize(EMU::BlobWriter& writer) {
    const auto kernel = VPUIP::createOrder3(kernel_sizeAttr());
    const auto strides = VPUIP::createOrder3(stridesAttr());
    const auto padsBegin = VPUIP::createOrder3(pads_beginAttr());
    const auto padsEnd = VPUIP::createOrder3(pads_endAttr());

    EMU::BlobWriter::String type;
    type = writer.createString("avg");

    const auto excludePad = writer.createString(exclude_pads() ? "true" : "false");

    MVCNN::PoolingParamsBuilder builder(writer);
    builder.add_pool_method(type);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_exclude_pad(excludePad);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PoolingParams});
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::AvgPoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(input());
    const auto pads = PadInfo(pads_begin(), pads_end());
    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, kernel_size(), strides(), pads);
    return inputTiling;
}

void vpux::VPU::AvgPoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    IE::adjustPaddings(this, inputTiling);
}
