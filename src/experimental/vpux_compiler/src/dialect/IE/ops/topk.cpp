
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

using namespace vpux;

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    auto kType = topK.k().getDefiningOp<mlir::ConstantOp>();
    if (kType == nullptr) {
        return mlir::failure();
    }

    const auto kArray = kType.value().dyn_cast<mlir::DenseElementsAttr>();
    if (kArray == nullptr) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(kArray.size() == 1, "K input must be scalar");

    const auto elementsRange = kArray.getValues<int64_t>();
    auto elementsIter = elementsRange.begin();

    mlir::SmallVector<int64_t, 4> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    outShape[topK.axis().getInt()] = *elementsIter;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.element_type().getValue());

    return mlir::success();
}
