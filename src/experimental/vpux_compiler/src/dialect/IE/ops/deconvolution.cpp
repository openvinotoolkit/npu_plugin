//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <ngraph/coordinate.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;
// Backprop to data is itself convolution, with inputs/outputs/attributes transmogrified as
// follows.
//
//                          Forward   Backward
// "N" axis for data batch  0         0
// "C" axis for data batch  1         1
// "Co" axis for filters    0         0
// "Ci" axis for filters    1         1
// "N" axis for output      0         0
// "C" axis for output      1         1
// Data batch               x         delta
// Data batch shape         S_x       S_o
// Filters                  f         reverse(f) [on spatial axes]
// Filters shape            S_f       S_f
// Window movement strides  q_x       p_x
// Window dilation strides  p_f       p_f
// Padding below            a_x       (S_f - 1)p_f - a_x
// Padding above            b_x       (S_f - 1)p_f +
//                                      + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f)
//                                         % q_x)
//                                      - b_x
// Data dilation strides    p_x       q_x
// Output shape             S_o       S_x
//
// To _validate_, we simply need to check/infer the output shape of the forward convolution,
// then check to make sure that the incoming delta has the same shape as the forward output.

mlir::LogicalResult vpux::IE::DeconvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));
    IE::DeconvolutionOpAdaptor convBackpropData(operands, attrs);
    if (mlir::failed(convBackpropData.verify(loc))) {
        return ::mlir::failure();
    }

    std::function<SmallVector<int64_t, MAX_NUM_DIMS>(mlir::ArrayAttr && arrayAttr)> convertArrayAttrToSmallVector =
            [](mlir::ArrayAttr&& arrayAttr) {
                SmallVector<int64_t, MAX_NUM_DIMS> result;
                for (auto&& a : arrayAttr)
                    result.push_back(a.dyn_cast<mlir::IntegerAttr>().getInt());
                return result;
            };

    auto featureShape = convBackpropData.feature().getType().cast<mlir::RankedTensorType>().getShape();
    auto featureType = convBackpropData.feature().getType().cast<mlir::RankedTensorType>().getElementType();
    auto outputShape = convBackpropData.output_shape();
    auto filterShape = convBackpropData.filter().getType().cast<mlir::RankedTensorType>().getShape();

    SmallVector<int64_t, MAX_NUM_DIMS> dataPaddingBelow = convertArrayAttrToSmallVector(convBackpropData.pads_end());
    SmallVector<int64_t, MAX_NUM_DIMS> dataPaddingAbove = convertArrayAttrToSmallVector(convBackpropData.pads_begin());
    SmallVector<int64_t, MAX_NUM_DIMS> windowStrides = convertArrayAttrToSmallVector(convBackpropData.strides());
    SmallVector<int64_t, MAX_NUM_DIMS> windowDilations = convertArrayAttrToSmallVector(convBackpropData.dilations());
    SmallVector<int64_t, MAX_NUM_DIMS> outputPadding = convertArrayAttrToSmallVector(convBackpropData.output_padding());

    if (outputShape != nullptr) {
        std::vector<ngraph::Dimension> nDataShape{std::next(featureShape.begin(), 2), featureShape.end()};
        std::vector<ngraph::Dimension> nFilterShape{std::next(filterShape.begin(), 2), filterShape.end()};

        auto outputShapeConstOp = outputShape.getDefiningOp<mlir::ConstantOp>();
        auto outputShapeDenseElementAttr = outputShapeConstOp.value().dyn_cast<mlir::DenseElementsAttr>();

        if (!outputShapeDenseElementAttr)
            return mlir::failure();

        auto elementRange = outputShapeDenseElementAttr.getValues<int64_t>();
        SmallVector<int64_t, MAX_NUM_DIMS> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[0]);
        mlirOutputShape.push_back(filterShape[1]);
        std::copy(elementRange.begin(), elementRange.end(), std::back_inserter(mlirOutputShape));
        inferredReturnShapes.emplace_back(mlirOutputShape, featureType);
    } else {
        std::vector<ngraph::Dimension> __resultShape;
        std::vector<ngraph::Dimension> nDataShape{std::next(featureShape.begin(), 2), featureShape.end()};
        std::vector<ngraph::Dimension> nFilterShape{std::next(filterShape.begin(), 2), filterShape.end()};

        ngraph::op::v1::ConvolutionBackpropData ngraph_op;
        ngraph_op.infer_conv_backprop_output_spatial_shape(
                nDataShape,                                                                // data_shape
                nFilterShape,                                                              // filter_sahpe
                ngraph::Strides(windowStrides.begin(), windowStrides.end()),               // strides
                ngraph::Strides(windowDilations.begin(), windowDilations.end()),           // dilations
                ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),  // pads_begin
                ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),  // pads_end
                ngraph::CoordinateDiff(outputPadding.begin(), outputPadding.end()),        // output_padding
                __resultShape);
        auto resultShape = ngraph::PartialShape{__resultShape}.get_shape();
        resultShape.insert(resultShape.begin(), filterShape[1]);
        resultShape.insert(resultShape.begin(), featureShape[0]);
        SmallVector<int64_t, MAX_NUM_DIMS> mlirOutputShape(resultShape.begin(), resultShape.end());
        inferredReturnShapes.emplace_back(mlirOutputShape, featureType);
    }
    return mlir::success();
}
