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

mlir::LogicalResult vpux::IE::ConcatOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));
    IE::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return ::mlir::failure();
    }

    auto inputSize = concat.input().size();
    auto inType = concat.input()[0].getType().cast<mlir::RankedTensorType>();
    auto inAxis = concat.axis().getInt();
    auto inShape = inType.getShape();
    auto inTypeSize = inShape.size();

    // Check: axis. Negative value means counting dimension from the end
    if (inAxis < 0) {
        inAxis += inTypeSize;
    }

    // set out shspe
    SmallVector<int64_t, 4> outShape(inTypeSize, 0);

    // init with first input
    auto inShapeIter = inShape.begin();
    for (auto outShapeIter = outShape.begin(); outShapeIter != outShape.end(); ++outShapeIter) {
        *outShapeIter = *inShapeIter;
        ++inShapeIter;
    }

    // concat with rest inputs
    for (uint32_t i = 1; i < inputSize; i++) {
        auto type = concat.input()[i].getType().cast<mlir::RankedTensorType>();
        auto shapeIter = type.getShape().begin();
        outShape[inAxis] += shapeIter[inAxis];
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    return mlir::success();
}
