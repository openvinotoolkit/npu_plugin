//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

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
