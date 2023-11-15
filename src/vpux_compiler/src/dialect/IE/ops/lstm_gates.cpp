//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LSTMGatesOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LSTMGatesOpAdaptor lstm(operands, attrs);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.initialCellState().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputHiddenState
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputCellState

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::LSTMGatesOp::verify() {
    const auto gatesInputShape = getShape(gatesInput()).raw();
    const auto initialCellStateShape = getShape(initialCellState()).raw();
    const auto batchSize = initialCellStateShape[0];
    const auto hiddenSize = initialCellStateShape[1];

    if (gatesInputShape != makeArrayRef<int64_t>({batchSize, 4 * hiddenSize})) {
        return errorAt(*this,
                       "Incompatible input shapes. Expected gatesInput shape: [batch_size, 4*hidden_size], "
                       "initialCellState shape: [batch_size, hidden_size]. Got gatesInput shape: {0}, initialCellState "
                       "shape: {1}",
                       gatesInputShape, initialCellStateShape);
    }

    return mlir::success();
}
