//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::VPU::ConvolutionOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ConvolutionOpAdaptor conv(operands, attrs);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = conv.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();
    const auto filterShape = conv.filter().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.dilations());

    static const auto ChanDim = Dim(1);
    if (inShape[ChanDim.ind()] != filterShape[ChanDim.ind()]) {
        return errorAt(loc, "Channels count of input tensor shape and filter shape must be the same: {0} != {1}",
                       inShape[ChanDim.ind()], filterShape[ChanDim.ind()]);
    }

    const auto outputShape =
            ngraph::infer_convolution_forward(EmptyNode::instance(), ngraph::Shape(inShape.begin(), inShape.end()),
                                              ngraph::Strides(windowStrides.size(), 1),  // dummy data dilations
                                              ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
                                              ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
                                              ngraph::Shape(filterShape.begin(), filterShape.end()),
                                              ngraph::Strides(windowStrides.begin(), windowStrides.end()),
                                              ngraph::Strides(windowDilations.begin(), windowDilations.end()));

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

EMU::BlobWriter::SpecificTask vpux::VPU::ConvolutionOp::serialize(EMU::BlobWriter& writer) {
    static const auto dY = Dim(2);
    static const auto dX = Dim(3);

    const auto strides = VPUIP::createOrder3(stridesAttr());
    const auto dilations = VPUIP::createOrder3(dilationsAttr());
    const auto padsBegin = VPUIP::createOrder3(pads_beginAttr());
    const auto padsEnd = VPUIP::createOrder3(pads_endAttr());

    const auto filterShape = getShape(filter());
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    MVCNN::SWConvolutionParamsBuilder builder(writer);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_dilations(&dilations);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_group(checked_cast<int32_t>(1));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SWConvolutionParams});
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::ConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto origInputShape = getShape(input());
    const auto origFilterShape = getShape(filter());
    const auto origBiasShape = bias() != nullptr ? getShape(bias()) : ShapeRef();
    const auto origPadding = PadInfo(pads_begin(), pads_end());

    return backInferConvTile(outputTile, origInputShape, origFilterShape, origBiasShape, strides(), origPadding);
}

void vpux::VPU::ConvolutionOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    IE::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::ConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
