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
#include "vpux/utils/core/small_vector.hpp"

#include <numeric>

#include <ngraph/op/strided_slice.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return ::mlir::failure();
    }

    std::function<ngraph::AxisSet(mlir::ArrayAttr && arrayAttr)> convertArrayAttrToAxisSet =
            [](mlir::ArrayAttr&& arrayAttr) {
                ngraph::AxisSet axis_set{};
                for (size_t i = 0; i < static_cast<size_t>(arrayAttr.size()); ++i) {
                    auto val = arrayAttr[i].dyn_cast<mlir::IntegerAttr>().getInt();
                    if (val == 1) {
                        axis_set.emplace(i);
                    }
                }
                return axis_set;
            };

    std::function<std::vector<int64_t>(mlir::ConstantOp)> convertConstantOpToVector = [](mlir::ConstantOp constOp) {
        auto denseElementArray = constOp.value().dyn_cast<mlir::DenseElementsAttr>();
        if (!denseElementArray)
            return std::vector<int64_t>();

        auto elementsRange = denseElementArray.getValues<int64_t>();
        return std::vector<int64_t>(elementsRange.begin(), elementsRange.end());
    };

    auto inDataType = slice.data().getType().cast<mlir::RankedTensorType>();
    auto inDataShape = inDataType.getShape();
    auto beginMask = convertArrayAttrToAxisSet(slice.begin_mask());
    auto endMask = convertArrayAttrToAxisSet(slice.end_mask());
    auto newAxisMask = convertArrayAttrToAxisSet(slice.new_axis_mask());
    auto shrinkAxisMask = convertArrayAttrToAxisSet(slice.shrink_axis_mask());
    auto ellipsisMask = convertArrayAttrToAxisSet(slice.ellipsis_mask());

    auto calculatedShape =
            ngraph::infer_slice_shape(nullptr, ngraph::Shape(inDataShape.begin(), inDataShape.end()),
                                      convertConstantOpToVector(slice.begin().getDefiningOp<mlir::ConstantOp>()),
                                      convertConstantOpToVector(slice.end().getDefiningOp<mlir::ConstantOp>()),
                                      convertConstantOpToVector(slice.stride().getDefiningOp<mlir::ConstantOp>()),
                                      beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    auto outputShape = calculatedShape.get_shape();
    SmallVector<int64_t, MAX_NUM_DIMS> mlirOutputShape(outputShape.begin(), outputShape.end());
    inferredReturnShapes.emplace_back(mlirOutputShape, inDataType.getElementType());
    return mlir::success();
}
