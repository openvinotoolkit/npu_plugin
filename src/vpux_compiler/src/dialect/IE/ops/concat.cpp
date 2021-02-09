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
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    const auto numInputs = concat.input().size();
    const auto inType = concat.input()[0].getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto inAxis = concat.axis().getInt();

    // Check: axis. Negative value means counting dimension from the end
    if (inAxis < 0) {
        inAxis += inRank;
    }

    // init with first input
    auto outShape = to_small_vector(inType.getShape());

    // concat with rest inputs
    for (size_t i = 1; i < numInputs; i++) {
        const auto curInType = concat.input()[i].getType().cast<mlir::ShapedType>();
        outShape[inAxis] += curInType.getShape()[inAxis];
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    return mlir::success();
}
