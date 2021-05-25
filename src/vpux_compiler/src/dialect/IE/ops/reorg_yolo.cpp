
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

mlir::LogicalResult vpux::IE::ReorgYoloOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReorgYoloOpAdaptor reorgYolo(operands, attrs);
    if (mlir::failed(reorgYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorgYolo.input().getType().cast<mlir::ShapedType>();

    if (reorgYolo.stride().getInt() <= 0) {
        return errorAt(loc, "Stride should be a natural number");
    }
    if (inType.getShape()[2] % reorgYolo.stride().getInt() != 0) {
        return errorAt(loc, "Input H should be divisible by stride.");
    }
    if (inType.getShape()[3] % reorgYolo.stride().getInt() != 0) {
        return errorAt(loc, "Input W should be divisible by stride.");
    }
    if (inType.getShape()[1] < reorgYolo.stride().getInt() * reorgYolo.stride().getInt()) {
        return errorAt(loc, "Input C >= (stride*stride) is required.");
    }

    SmallVector<int64_t> outputShape{inType.getShape()[0], inType.getShape()[1]};
    for (size_t i = 2; i < inType.getShape().size(); i++) {
        outputShape.push_back(inType.getShape()[i] / reorgYolo.stride().getInt());
        outputShape[1] *= reorgYolo.stride().getInt();
    }

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());
    return mlir::success();
}
