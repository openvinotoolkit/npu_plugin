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

mlir::LogicalResult vpux::IE::ReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReshapeOpAdaptor reshape(operands, attrs);
    if (mlir::failed(reshape.verify(loc))) {
        return ::mlir::failure();
    }

    auto specialZero = reshape.special_zero();
    auto inDataType = reshape.input1().getType().cast<mlir::RankedTensorType>();
    auto inDataShape = inDataType.getShape();
    auto inShape = reshape.input2().getDefiningOp<mlir::ConstantOp>();

    if (inShape) {
        auto denseElementArray = inShape.value().dyn_cast<mlir::DenseElementsAttr>();
        if (!denseElementArray)
            return mlir::failure();

        auto elementsRange = denseElementArray.getValues<int64_t>();
        std::vector<int64_t> outShapeVec(elementsRange.begin(), elementsRange.end());
        const auto zeroDims = std::count_if(outShapeVec.begin(), outShapeVec.end(), [](int64_t v) {
            return v == 0;
        });
        const auto negativeDims = std::count_if(outShapeVec.begin(), outShapeVec.end(), [](int64_t v) {
            return v == -1;
        });

        if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
            inferredReturnShapes.emplace_back(makeArrayRef(outShapeVec), inDataType.getElementType());
            return mlir::success();
        } else {
            std::vector<int64_t> inDataShapeVec(inDataShape.begin(), inDataShape.end());
            int64_t dividend =
                    std::accumulate(inDataShape.begin(), inDataShape.end(), int64_t(1), std::multiplies<int64_t>());

            for (size_t i = 0; i < outShapeVec.size(); ++i) {
                auto& v = outShapeVec[i];
                if (v == 0 && specialZero) {
                    if (v < static_cast<int64_t>(inDataShapeVec.size())) {
                        v = inDataShapeVec[i];
                    } else {
                        return mlir::failure();
                    }
                }
                if (v > 0) {
                    auto remainder = dividend % v;
                    if (remainder != 0) {
                        return mlir::failure();
                    }
                    dividend /= v;
                }
            }

            if (negativeDims > 0) {
                auto negIt = std::find(outShapeVec.begin(), outShapeVec.end(), -1);
                if (negIt != outShapeVec.end()) {
                    *negIt = dividend;
                }
            }

            auto outShape = makeArrayRef(outShapeVec);
            inferredReturnShapes.emplace_back(outShape, inDataType.getElementType());
            return mlir::success();
        }
    }

    return mlir::failure();
}
