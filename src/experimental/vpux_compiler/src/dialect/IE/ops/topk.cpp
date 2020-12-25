
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
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return ::mlir::failure();
    }

    auto inType = topK.input().getType().cast<mlir::RankedTensorType>();
    auto kType = topK.k().getDefiningOp<mlir::ConstantOp>();

    if (kType) {
        auto kArray = kType.value().dyn_cast<mlir::DenseElementsAttr>();
        VPUX_THROW_UNLESS(static_cast<size_t>(kArray.size()) == 1, "K input must be scalar");
        if (kArray) {
            auto elementsRange = kArray.getValues<int64_t>();

            auto elementsIter = elementsRange.begin();
            if (elementsIter == elementsRange.end()) {
                return mlir::LogicalResult(printTo(mlir::emitError(loc), "TopK: 'axis' tensor is empty"));
            }
            auto inputShape = inType.getShape().vec();
            mlir::SmallVector<int64_t, 4> outShape;
            for (size_t i = 0; i < inputShape.size(); ++i) {
                outShape.emplace_back(inputShape[i]);
            }
            outShape[topK.axis().getInt()] = checked_cast<int64_t>(*elementsIter);

            inferredReturnShapes.emplace_back(outShape, inType.getElementType());

            inferredReturnShapes.emplace_back(outShape, topK.element_type().getValue());
            return mlir::success();
        }
    }
    return mlir::LogicalResult(printTo(mlir::emitError(loc), "TopK: 'axis' input must be produced by ConstantOp"));
}
