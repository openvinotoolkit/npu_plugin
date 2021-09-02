
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

mlir::LogicalResult vpux::IE::BroadcastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::BroadcastOpAdaptor broadcast(operands, attrs);
    if (mlir::failed(broadcast.verify(loc))) {
        return mlir::failure();
    }

    // SmallVector<int64_t> outShape = {2, 3, 6};
    const auto targetShape = broadcast.target_shape().cast<mlir::ShapedType>().getShape().;
    // SmallVector<int64_t> outShape = targetShape.raw();
    // std::cout << outShape[0] << outShape[1];

    const auto inType = broadcast.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(targetShape, inType.getElementType());

    return mlir::success();
}
