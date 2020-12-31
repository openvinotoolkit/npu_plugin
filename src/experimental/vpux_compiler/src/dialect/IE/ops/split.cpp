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

mlir::LogicalResult vpux::IE::SplitOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SplitOpAdaptor split(operands, attrs);
    if (mlir::failed(split.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = split.input().getType().cast<mlir::ShapedType>();

    auto inAxis = split.axis().getDefiningOp<mlir::ConstantOp>();
    if (inAxis == nullptr) {
        return mlir::failure();
    }

    auto denseElementArray = inAxis.value().dyn_cast<mlir::DenseElementsAttr>();
    if (denseElementArray == nullptr) {
        return mlir::failure();
    }

    const auto elementsRange = denseElementArray.getValues<int64_t>();

    auto elementsIter = elementsRange.begin();
    if (elementsIter == elementsRange.end()) {
        return mlir::failure();
    }

    auto outShape = to_small_vector(inType.getShape());
    if (outShape[*elementsIter] < split.num_splits().getInt() ||
        outShape[*elementsIter] % split.num_splits().getInt() != 0) {
        return mlir::failure();
    }
    outShape[*elementsIter] /= split.num_splits().getInt();

    for (int i = 0; i < split.num_splits().getInt(); ++i) {
        inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    }

    return mlir::success();
}
