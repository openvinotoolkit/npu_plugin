// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ExpandOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ExpandOpAdaptor expand(operands, attrs);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr(expand.pads_begin_attr());
    const auto padEnd = parseIntArrayAttr(expand.pads_end_attr());

    const auto inType = expand.input().getType().cast<mlir::ShapedType>();
    if (!inType) {
        return mlir::failure();
    }

    const auto inputShape = inType.getShape();
    SmallVector<int64_t> outShape(inputShape.size());
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape[i] = padBegin[i] + inputShape[i] + padEnd[i];
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
