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

mlir::LogicalResult vpux::IE::UnsqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::UnsqueezeOpAdaptor unsqueeze(operands, attrs);
    if (mlir::failed(unsqueeze.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = unsqueeze.input1().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    auto inAxes = unsqueeze.input2().getDefiningOp<mlir::ConstantOp>();
    if (inAxes == nullptr) {
        return mlir::failure();
    }

    const auto denseElementArray = inAxes.value().dyn_cast<mlir::DenseElementsAttr>();
    if (denseElementArray == nullptr) {
        return mlir::failure();
    }

    auto axesVec = to_small_vector(denseElementArray.getValues<int64_t>());
    std::sort(axesVec.begin(), axesVec.end());

    SmallVector<int64_t> outShapeVec(inDataShape.begin(), inDataShape.end());
    const auto outRank = static_cast<int64_t>(outShapeVec.size() + axesVec.size());

    if (*axesVec.rbegin() >= outRank) {
        return mlir::failure();
    }

    for (auto a : axesVec) {
        outShapeVec.insert(outShapeVec.begin() + a, 1);
    }

    inferredReturnShapes.emplace_back(makeArrayRef(outShapeVec), inDataType.getElementType());
    return mlir::success();
}
