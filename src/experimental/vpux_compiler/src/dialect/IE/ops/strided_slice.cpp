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
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <numeric>

#include <ngraph/op/strided_slice.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ngraph::AxisSet axis_set;

        const auto arr = parseIntArrayAttr(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axis_set.emplace(p.index());
            }
        }

        return axis_set;
    };

    const auto convertConstantOpToVector = [](mlir::ConstantOp constOp) -> std::vector<int64_t> {
        if (constOp == nullptr) {
            return {};
        }

        const auto denseElementArray = constOp.value().dyn_cast<mlir::DenseElementsAttr>();
        if (denseElementArray == nullptr) {
            return {};
        }

        return to_std_vector(denseElementArray.getValues<int64_t>());
    };

    const auto inDataType = slice.data().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();
    const auto beginMask = getAxisSetArr(slice.begin_mask());
    const auto endMask = getAxisSetArr(slice.end_mask());
    const auto newAxisMask = getAxisSetArr(slice.new_axis_mask());
    const auto shrinkAxisMask = getAxisSetArr(slice.shrink_axis_mask());
    const auto ellipsisMask = getAxisSetArr(slice.ellipsis_mask());

    const auto outputShape =
            ngraph::infer_slice_shape(nullptr, ngraph::Shape(inDataShape.begin(), inDataShape.end()),
                                      convertConstantOpToVector(slice.begin().getDefiningOp<mlir::ConstantOp>()),
                                      convertConstantOpToVector(slice.end().getDefiningOp<mlir::ConstantOp>()),
                                      convertConstantOpToVector(slice.stride().getDefiningOp<mlir::ConstantOp>()),
                                      beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto shapeI64 = to_vector<4>(outputShape.get_shape() | transformed([](size_t val) {
                                           return checked_cast<int64_t>(val);
                                       }));
    inferredReturnShapes.emplace_back(shapeI64, inDataType.getElementType());

    return mlir::success();
}
