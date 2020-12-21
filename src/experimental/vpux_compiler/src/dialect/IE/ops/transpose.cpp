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

using namespace vpux;

mlir::LogicalResult vpux::IE::TransposeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TransposeOpAdaptor transpose(operands, attrs);
    if (mlir::failed(transpose.verify(loc))) {
        return ::mlir::failure();
    }

    auto inDataType = transpose.input1().getType().cast<mlir::RankedTensorType>();
    auto inDataShape = inDataType.getShape();
    auto inOrder = transpose.input2().getDefiningOp<mlir::ConstantOp>();
    std::vector<int64_t> outShapeVec(inDataShape.begin(), inDataShape.end());

    if (inOrder) {
        auto denseElementArray = inOrder.value().dyn_cast<mlir::DenseElementsAttr>();
        if (!denseElementArray)
            return mlir::failure();

        auto elementsRange = denseElementArray.getValues<int64_t>();
        std::vector<int64_t> inOrderVec(elementsRange.begin(), elementsRange.end());

        if (inOrderVec.size() == 0) {
            std::reverse(outShapeVec.begin(), outShapeVec.end());
        } else {
            if (outShapeVec.size() != inOrderVec.size()) {
                return mlir::failure();
            }

            int64_t outRank = static_cast<int64_t>(outShapeVec.size());
            std::vector<int64_t> tmp(outShapeVec.begin(), outShapeVec.end());
            for (size_t i = 0; i < inOrderVec.size(); ++i) {
                if (inOrderVec[i] >= outRank) {
                    return mlir::failure();
                }
                outShapeVec[i] = tmp[inOrderVec[i]];
            }
        }

        inferredReturnShapes.emplace_back(makeArrayRef(outShapeVec), inDataType.getElementType());
        return mlir::success();
    }

    return mlir::failure();
}
