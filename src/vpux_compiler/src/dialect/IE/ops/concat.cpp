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

//
// getAxis
//

namespace {

Dim normalizeAxis(IE::ConcatOpAdaptor concat) {
    const auto inType = concat.inputs().front().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto axisInd = concat.axis().getSInt();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Concat axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

}  // namespace

Dim vpux::IE::ConcatOp::getAxis() {
    return normalizeAxis(*this);
}

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ConcatOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ConcatOpAdaptor concat(operands, attrs);
    if (mlir::failed(concat.verify(loc))) {
        return mlir::failure();
    }

    const auto axis = normalizeAxis(concat);

    // Init with first input
    auto outShape = getShape(concat.inputs().front()).toValues();

    // Concat with rest inputs
    for (auto i : irange<size_t>(1, concat.inputs().size())) {
        const auto curShape = getShape(concat.inputs()[i]);
        outShape[axis] += curShape[axis];
    }

    const auto elemType = concat.inputs().front().getType().cast<mlir::ShapedType>().getElementType();
    inferredReturnShapes.emplace_back(outShape.raw(), elemType);

    return mlir::success();
}
