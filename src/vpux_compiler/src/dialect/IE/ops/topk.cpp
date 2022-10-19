//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    auto kConst = topK.k().getDefiningOp<Const::DeclareOp>();
    if (kConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for k");
    }

    const auto kContent = kConst.content();
    if (!kContent.isSplat()) {
        return errorAt(loc, "K input must be scalar");
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.axis().getInt();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }

    outShape[axis] = kContent.getSplatValue<int64_t>();

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.element_type().getValue());

    return mlir::success();
}

//
// inferLayoutInfo
//

void vpux::IE::TopKOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto inOrder = info.getInput(0);

    info.setInput(0, inOrder);
    info.setOutput(0, inOrder);
    info.setOutput(1, inOrder);
}
