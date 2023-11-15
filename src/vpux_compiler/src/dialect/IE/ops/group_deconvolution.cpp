//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/group_conv.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::GroupDeconvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupDeconvolutionOpAdaptor groupDeconv(operands, attrs);
    if (mlir::failed(groupDeconv.verify(loc))) {
        return mlir::failure();
    }

    const auto featureType = groupDeconv.feature().getType().cast<NDTypeInterface>();
    const auto featureShape = to_small_vector(featureType.getShape());
    const auto featureElemType = featureType.getElementType();
    const auto outputShape = groupDeconv.output_shape();
    const auto filterShape = to_small_vector(groupDeconv.filter().getType().cast<NDTypeInterface>().getShape());

    if ((filterShape.size() != 5) || (featureShape.size() != 4)) {
        return errorAt(loc,
                       "Only supported filterShape.size() == 5 && featureShape.size() == 4, but filter has size {0} "
                       "shape and feature has size {1} shape",
                       filterShape.size(), featureShape.size());
    }

    if (outputShape != nullptr) {
        return errorAt(loc, "Explicit output shape is not implemented");
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(groupDeconv.pads_end());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(groupDeconv.pads_begin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(groupDeconv.strides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(groupDeconv.dilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(groupDeconv.output_padding());

    // For 2D GroupDeconvolution:
    // input tensor layout is [N, C_IN * GROUPS, H, W]
    // kernel tensor layout is [GROUPS, C_IN, C_OUT, kH, kW]
    const std::vector<ngraph::Dimension> nDataShape(std::next(featureShape.begin(), 2), featureShape.end());
    const std::vector<ngraph::Dimension> nFilterShape(std::next(filterShape.begin(), 3), filterShape.end());

    ngraph::op::v1::GroupConvolutionBackpropData ngraph_op;
    std::vector<ngraph::Dimension> __resultShape;
    ngraph_op.infer_conv_backprop_output_spatial_shape(
            nDataShape,                                                                // data_shape
            nFilterShape,                                                              // filter_shape
            ngraph::Strides(windowStrides.begin(), windowStrides.end()),               // strides
            ngraph::Strides(windowDilations.begin(), windowDilations.end()),           // dilations
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),  // pads_begin
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),  // pads_end
            ngraph::CoordinateDiff(outputPadding.begin(), outputPadding.end()),        // output_padding
            __resultShape);
    const auto resultShape = ngraph::PartialShape{std::move(__resultShape)}.get_shape();

    SmallVector<int64_t> mlirOutputShape;
    auto groups = filterShape[0];
    auto OC = filterShape[2];
    mlirOutputShape.push_back(featureShape[0]);
    mlirOutputShape.push_back(groups * OC);
    std::copy(resultShape.begin(), resultShape.end(), std::back_inserter(mlirOutputShape));

    inferredReturnShapes.emplace_back(mlirOutputShape, featureElemType);

    return mlir::success();
}
