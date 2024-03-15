//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <openvino/op/convolution.hpp>

using namespace vpux;

mlir::LogicalResult vpux::VPU::TransposedConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TransposedConvolutionOpAdaptor convBackpropData(operands, attrs);
    if (mlir::failed(convBackpropData.verify(loc))) {
        return mlir::failure();
    }

    const auto featureType = convBackpropData.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto featureShape = featureType.getShape().raw();
    const auto outputShape = convBackpropData.getOutputShape();
    const auto filterShape = convBackpropData.getFilter().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(convBackpropData.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(convBackpropData.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(convBackpropData.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(convBackpropData.getDilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(convBackpropData.getOutputPadding());

    if (outputShape != nullptr) {
        const SmallVector<ov::Dimension> nDataShape(std::next(featureShape.begin(), 2), featureShape.end());
        const SmallVector<ov::Dimension> nFilterShape(std::next(filterShape.begin(), 2), filterShape.end());

        auto outputShapeConst = outputShape.getDefiningOp<Const::DeclareOp>();
        if (outputShapeConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for output_shape");
        }

        const auto outputShapeContent = outputShapeConst.getContent();
        const auto outputShapeVals = outputShapeContent.getValues<int64_t>();

        SmallVector<int64_t> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[Dims4D::Act::N.ind()]);
        mlirOutputShape.push_back(filterShape[Dims4D::Filter::OC.ind()]);
        std::copy(outputShapeVals.begin(), outputShapeVals.end(), std::back_inserter(mlirOutputShape));

        auto outType = featureType.changeShape(Shape(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    } else {
        const std::vector<ov::Dimension> nDataShape(std::next(featureShape.begin(), 2), featureShape.end());
        const std::vector<ov::Dimension> nFilterShape(std::next(filterShape.begin(), 2), filterShape.end());

        ov::op::v1::ConvolutionBackpropData ov_op;
        std::vector<ov::Dimension> spatialShape;
        ov_op.infer_conv_backprop_output_spatial_shape(
                nDataShape,                                                            // data_shape
                nFilterShape,                                                          // filter_sahpe
                ov::Strides(windowStrides.begin(), windowStrides.end()),               // strides
                ov::Strides(windowDilations.begin(), windowDilations.end()),           // dilations
                ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),  // pads_begin
                ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),  // pads_end
                ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()),        // output_padding
                spatialShape);
        const auto resultShape = ov::PartialShape(std::move(spatialShape)).get_shape();

        SmallVector<int64_t> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[Dims4D::Act::N.ind()]);
        mlirOutputShape.push_back(filterShape[Dims4D::Filter::OC.ind()]);
        std::copy(resultShape.begin(), resultShape.end(), std::back_inserter(mlirOutputShape));

        auto outType = featureType.changeShape(Shape(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}
