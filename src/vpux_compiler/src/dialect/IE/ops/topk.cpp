//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::TopKOp::verify() {
    auto kNumElements = k().getType().cast<vpux::NDTypeInterface>().getNumElements();
    if (kNumElements != 1) {
        return errorAt(*this, "K should have only 1 element, while it has {0}", kNumElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    auto reshapeK = topK.k().getDefiningOp<IE::ReshapeOp>();
    auto kConst = (reshapeK != nullptr) ? reshapeK.input().getDefiningOp<Const::DeclareOp>()
                                        : topK.k().getDefiningOp<Const::DeclareOp>();
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
    int64_t axis = topK.axis();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }

    outShape[axis] = kContent.getSplatValue<int64_t>();

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.element_type());

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
