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

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GatherOpAdaptor gather(operands, attrs);
    if (mlir::failed(gather.verify(loc))) {
        return ::mlir::failure();
    }

    auto inType = gather.input().getType().cast<mlir::RankedTensorType>();
    auto inIndices = gather.indices().getType().cast<mlir::RankedTensorType>();
    auto inAxis = gather.axis().getDefiningOp<mlir::ConstantOp>();

    std::vector<int64_t> outShape;
    outShape.reserve(inType.getShape().size() + inIndices.getShape().size() - 1);

    if (inAxis) {
        auto denseElementArray = inAxis.value().dyn_cast<mlir::DenseElementsAttr>();
        if (denseElementArray) {
            auto elementsRange = denseElementArray.getValues<int64_t>();

            auto elementsIter = elementsRange.begin();
            if (elementsIter == elementsRange.end()) {
                return ::mlir::failure();
            }
            auto inputShape = inType.getShape().vec();
            // calculate output shapes
            for (size_t i = 0; i < inputShape.size(); ++i) {
                if (i == checked_cast<size_t>(*elementsIter)) {
                    auto indicesShape = inIndices.getShape().vec();
                    for (size_t j = 0; j < indicesShape.size(); ++j) {
                        outShape.push_back(indicesShape[j]);
                    }
                } else {
                    outShape.push_back(inputShape[i]);
                }
            }
        }
    }
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

SmallVector<mlir::Value, 4> vpux::IE::GatherOp::getInputs() {
    return {input()};
}

SmallVector<mlir::Value, 1> vpux::IE::GatherOp::getOutputs() {
    return {output()};
}
