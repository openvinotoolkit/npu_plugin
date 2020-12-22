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

#include <ngraph/type/float16.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::SoftMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SoftMaxOpAdaptor softMax(operands, attrs);
    if (mlir::failed(softMax.verify(loc))) {
        return ::mlir::failure();
    }

    auto inType = softMax.input().getType().cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::SoftMaxOp::fold(ArrayRef<mlir::Attribute>) {
    auto shape = input().getType().cast<mlir::ShapedType>();
    auto numElements = shape.getNumElements();
    auto axis = axisInd();

    VPUX_THROW_UNLESS(axis < shape.getShape().size(), "Wrong axis idx {0} for {1} dim tensor", axis,
                      shape.getShape().size());
    if (shape.getShape()[axis] > 1L) {
        return nullptr;
    }

    mlir::DenseElementsAttr denseAttrOfOnes;
    if (shape.getElementType().isF32()) {
        std::vector<float> arrayOfOnes(numElements, 1.0f);
        denseAttrOfOnes = mlir::DenseElementsAttr::get(shape, makeArrayRef(arrayOfOnes.data(), numElements));
    } else if (shape.getElementType().isF16()) {
        std::vector<ngraph::float16> arrayOfOnes(numElements, 1.0f);
        denseAttrOfOnes = mlir::DenseElementsAttr::get(shape, makeArrayRef(arrayOfOnes.data(), numElements));
    }
    return denseAttrOfOnes;
}
