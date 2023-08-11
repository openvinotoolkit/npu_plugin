//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::VPU::DeconvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DeconvolutionOpAdaptor convBackpropData(operands, attrs);
    if (mlir::failed(convBackpropData.verify(loc))) {
        return mlir::failure();
    }

    const auto featureType = convBackpropData.feature().getType().cast<vpux::NDTypeInterface>();
    const auto featureShape = featureType.getShape().raw();
    const auto outputShape = convBackpropData.output_shape();
    const auto filterShape = convBackpropData.filter().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(convBackpropData.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(convBackpropData.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(convBackpropData.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(convBackpropData.dilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(convBackpropData.output_padding());

    if (outputShape != nullptr) {
        const SmallVector<ngraph::Dimension> nDataShape(std::next(featureShape.begin(), 2), featureShape.end());
        const SmallVector<ngraph::Dimension> nFilterShape(std::next(filterShape.begin(), 2), filterShape.end());

        auto outputShapeConst = outputShape.getDefiningOp<Const::DeclareOp>();
        if (outputShapeConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for output_shape");
        }

        const auto outputShapeContent = outputShapeConst.content();
        const auto outputShapeVals = outputShapeContent.getValues<int64_t>();

        SmallVector<int64_t> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[0]);
        mlirOutputShape.push_back(filterShape[1]);
        std::copy(outputShapeVals.begin(), outputShapeVals.end(), std::back_inserter(mlirOutputShape));

        auto outType = featureType.changeShape(Shape(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    } else {
        const std::vector<ngraph::Dimension> nDataShape(std::next(featureShape.begin(), 2), featureShape.end());
        const std::vector<ngraph::Dimension> nFilterShape(std::next(filterShape.begin(), 2), filterShape.end());

        ngraph::op::v1::ConvolutionBackpropData ngraph_op;
        std::vector<ngraph::Dimension> __resultShape;
        ngraph_op.infer_conv_backprop_output_spatial_shape(
                nDataShape,                                                                // data_shape
                nFilterShape,                                                              // filter_sahpe
                ngraph::Strides(windowStrides.begin(), windowStrides.end()),               // strides
                ngraph::Strides(windowDilations.begin(), windowDilations.end()),           // dilations
                ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),  // pads_begin
                ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),  // pads_end
                ngraph::CoordinateDiff(outputPadding.begin(), outputPadding.end()),        // output_padding
                __resultShape);
        const auto resultShape = ngraph::PartialShape{__resultShape}.get_shape();

        SmallVector<int64_t> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[0]);
        mlirOutputShape.push_back(filterShape[1]);
        std::copy(resultShape.begin(), resultShape.end(), std::back_inserter(mlirOutputShape));

        auto outType = featureType.changeShape(Shape(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::DeconvolutionOp::serialize(EMU::BlobWriter& writer) {
    static const auto dY = Dim(Dims4D::Filter::KY);
    static const auto dX = Dim(Dims4D::Filter::KX);

    const auto strides = VPUIP::createOrder3(stridesAttr());
    const auto dilations = VPUIP::createOrder3(dilationsAttr());
    const auto padsBegin = VPUIP::createOrder3(pads_beginAttr());
    const auto padsEnd = VPUIP::createOrder3(pads_endAttr());
    const auto outputPadding = VPUIP::createOrder3(output_paddingAttr());

    const auto filterShape = getShape(filter());
    const auto kernel =
            MVCNN::order3(checked_cast<uint8_t>(filterShape[dX]), checked_cast<uint8_t>(filterShape[dY]), 0);

    MVCNN::DeconvolutionParamsBuilder builder(writer);
    builder.add_kernel(&kernel);
    builder.add_strides(&strides);
    builder.add_dilations(&dilations);
    builder.add_pads_begin(&padsBegin);
    builder.add_pads_end(&padsEnd);
    builder.add_output_padding(&outputPadding);
    builder.add_is_depthwise(false);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_DeconvolutionParams});
}
