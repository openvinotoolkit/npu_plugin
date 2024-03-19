//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <openvino/op/group_conv.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::GroupTransposedConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupTransposedConvolutionOpAdaptor groupTransposedConv(operands, attrs);
    if (mlir::failed(groupTransposedConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = groupTransposedConv.getInput().getType().cast<NDTypeInterface>();
    const auto inputShape = to_small_vector(inputType.getShape());
    const auto inputElemType = inputType.getElementType();
    const auto outputShape = groupTransposedConv.getOutputShape();
    const auto filterShape =
            to_small_vector(groupTransposedConv.getFilter().getType().cast<NDTypeInterface>().getShape());

    if (outputShape != nullptr) {
        return errorAt(loc, "Explicit output shape is not implemented");
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(groupTransposedConv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(groupTransposedConv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(groupTransposedConv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(groupTransposedConv.getDilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(groupTransposedConv.getOutputPadding());

    // For 2D GroupTransposedConvolution:
    // input tensor layout is [N, C_IN * GROUPS, H, W]
    // kernel tensor layout is [GROUPS, C_OUT, C_IN, kH, kW]
    const std::vector<ov::Dimension> nDataShape(std::next(inputShape.begin(), 2), inputShape.end());
    const std::vector<ov::Dimension> nFilterShape(std::next(filterShape.begin(), 3), filterShape.end());

    ov::op::v1::GroupConvolutionBackpropData ov_op;
    std::vector<ov::Dimension> __resultShape;
    ov_op.infer_conv_backprop_output_spatial_shape(
            nDataShape,                                                            // data_shape
            nFilterShape,                                                          // filter_shape
            ov::Strides(windowStrides.begin(), windowStrides.end()),               // strides
            ov::Strides(windowDilations.begin(), windowDilations.end()),           // dilations
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),  // pads_begin
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),  // pads_end
            ov::CoordinateDiff(outputPadding.begin(), outputPadding.end()),        // output_padding
            __resultShape);
    const auto resultShape = ov::PartialShape{std::move(__resultShape)}.get_shape();

    SmallVector<int64_t> mlirOutputShape;
    auto groups = filterShape[IE::GROUP_TRANSPOSED_CONV_GROUPS_DIM_INDEX];
    auto OC = filterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX];
    mlirOutputShape.push_back(inputShape[Dims4D::Act::N.ind()]);
    mlirOutputShape.push_back(groups * OC);
    std::copy(resultShape.begin(), resultShape.end(), std::back_inserter(mlirOutputShape));

    inferredReturnShapes.emplace_back(mlirOutputShape, inputElemType);

    return mlir::success();
}
