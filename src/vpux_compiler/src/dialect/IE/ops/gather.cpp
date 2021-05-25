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

mlir::LogicalResult vpux::IE::GatherOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GatherOpAdaptor gather(operands, attrs);
    if (mlir::failed(gather.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gather.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();
    const auto indicesShape = gather.indices().getType().cast<mlir::ShapedType>().getShape();

    auto axisConst = gather.axis().getDefiningOp<ConstantInterface>();
    if (axisConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for axis");
    }

    const auto axis = axisConst.getContent().getValues<int64_t>()[0];

    SmallVector<int64_t> outShape;
    outShape.reserve(inputShape.size() + indicesShape.size() - 1);

    // calculate output shapes
    for (size_t i = 0; i < inputShape.size(); ++i) {
        if (i == checked_cast<size_t>(axis)) {
            for (size_t j = 0; j < indicesShape.size(); ++j) {
                outShape.push_back(indicesShape[j]);
            }
        } else {
            outShape.push_back(inputShape[i]);
        }
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
