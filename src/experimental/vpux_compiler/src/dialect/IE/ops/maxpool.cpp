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

#include <mlir/IR/PatternMatch.h>

#include <ngraph/coordinate.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs);
    if (mlir::failed(maxPool.verify(loc))) {
        return ::mlir::failure();
    }

    /* FIXME: this can be a utility function to be used across operations */
    std::function<SmallVector<int64_t, MAX_NUM_DIMS>(mlir::ArrayAttr && arrayAttr)> convertArrayAttrToSmallVector =
            [](mlir::ArrayAttr&& arrayAttr) {
                SmallVector<int64_t, MAX_NUM_DIMS> result;
                for (auto&& a : arrayAttr)
                    result.push_back(a.dyn_cast<mlir::IntegerAttr>().getInt());
                return result;
            };

    SmallVector<int64_t, MAX_NUM_DIMS> dataPaddingBelow = convertArrayAttrToSmallVector(maxPool.pads_end());
    SmallVector<int64_t, MAX_NUM_DIMS> dataPaddingAbove = convertArrayAttrToSmallVector(maxPool.pads_begin());
    SmallVector<int64_t, MAX_NUM_DIMS> windowShape = convertArrayAttrToSmallVector(maxPool.kernel_size());
    SmallVector<int64_t, MAX_NUM_DIMS> windowStrides = convertArrayAttrToSmallVector(maxPool.strides());
    SmallVector<int64_t, MAX_NUM_DIMS> dataShape;
    auto kernelDim = windowShape.size();
    auto roundingType = maxPool.rounding_type().getValue();

    auto inType = maxPool.input().getType().cast<mlir::RankedTensorType>();

    std::copy(inType.getShape().end() - kernelDim, inType.getShape().end(), std::back_inserter(dataShape));

    auto outputShape = ngraph::infer_windowed_reduction_output_shape(
            nullptr, ngraph::Shape(dataShape.begin(), dataShape.end()),
            ngraph::Strides(kernelDim, 1) /* dilation is not applicable in maxpooling */,
            ngraph::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ngraph::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ngraph::Shape(windowShape.begin(), windowShape.end()),
            ngraph::Strides(windowStrides.begin(), windowStrides.end()),
            ngraph::Strides(kernelDim, 1) /* dilation is not applicable in maxpooling */, true,
            roundingType == vpux::IE::RoundingType::CEIL);

    // Make __outputShape to have same rank as inType
    SmallVector<int64_t, MAX_NUM_DIMS> __outputShape;

    std::copy(inType.getShape().begin(), inType.getShape().end() - kernelDim, std::back_inserter(__outputShape));

    for (auto i = 0; i < outputShape.rank().get_length(); i++)
        __outputShape.push_back(outputShape[i].get_length());

    inferredReturnShapes.emplace_back(__outputShape, inType.getElementType());

    return mlir::success();
}
