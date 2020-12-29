
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
        return mlir::LogicalResult(printTo(mlir::emitError(loc), "Stride should be a natural number"));
    }
    if (inType.getShape()[2] % reorgYolo.stride().getInt() != 0) {
        return mlir::LogicalResult(
                printTo(mlir::emitError(loc), "For [N, C, H, W] input shape, H should be divisible by stride."));
    }
    if (inType.getShape()[3] % reorgYolo.stride().getInt() != 0) {
        return mlir::LogicalResult(
                printTo(mlir::emitError(loc), "For [N, C, H, W] input shape, W should be divisible by stride."));
    }
    if (inType.getShape()[1] < reorgYolo.stride().getInt() * reorgYolo.stride().getInt()) {
        return mlir::LogicalResult(
                printTo(mlir::emitError(loc), "For [N, C, H, W] input shape, C >= (stride*stride) is required."));
    }

    mlir::SmallVector<int64_t, 4> outputShape{inType.getShape()[0], inType.getShape()[1]};
    for (size_t i = 2; i < inType.getShape().size(); i++) {
        outputShape.push_back(inType.getShape()[i] / reorgYolo.stride().getInt());
        outputShape[1] *= reorgYolo.stride().getInt();
    }

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());
    return mlir::success();
}
