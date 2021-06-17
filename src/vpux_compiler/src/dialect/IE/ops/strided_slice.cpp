//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

    auto beginConst = slice.begin().getDefiningOp<ConstantInterface>();
    auto endConst = slice.end().getDefiningOp<ConstantInterface>();
    auto strideConst = slice.stride().getDefiningOp<ConstantInterface>();
    if (beginConst == nullptr || endConst == nullptr || strideConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for begin, end and stride");
    }

    const auto begin = to_std_vector(beginConst.getContent().getValues<int64_t>());
    const auto end = to_std_vector(endConst.getContent().getValues<int64_t>());
    const auto stride = to_std_vector(strideConst.getContent().getValues<int64_t>());

    const auto inDataType = slice.data().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();
    const auto beginMask = getAxisSetArr(slice.begin_mask());
    const auto endMask = getAxisSetArr(slice.end_mask());
    const auto newAxisMask = getAxisSetArr(slice.new_axis_mask());
    const auto shrinkAxisMask = getAxisSetArr(slice.shrink_axis_mask());
    const auto ellipsisMask = getAxisSetArr(slice.ellipsis_mask());

    const auto outputShape =
            ngraph::infer_slice_shape(nullptr, ngraph::Shape(inDataShape.begin(), inDataShape.end()), begin, end,
                                      stride, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    auto log = vpux::Logger::global();
    log.error("shapeI64 {0}", shapeI64);
    inferredReturnShapes.emplace_back(shapeI64, inDataType.getElementType());

    return mlir::success();
}
