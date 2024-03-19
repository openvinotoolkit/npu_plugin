//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMGatesOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LSTMGatesOpAdaptor lstm(operands, attrs);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.getInitialCellState().getType().cast<vpux::NDTypeInterface>();

    inferredReturnTypes.push_back(inType);  // outputHiddenState
    inferredReturnTypes.push_back(inType);  // outputCellState

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::LSTMGatesOp::verify() {
    const auto gatesInputShape = getShape(getGatesInput()).raw();
    const auto initialCellStateShape = getShape(getInitialCellState()).raw();
    const auto batchSize = initialCellStateShape[0];
    const auto hiddenSize = initialCellStateShape[1];

    if (gatesInputShape != ArrayRef<int64_t>({batchSize, 4 * hiddenSize})) {
        return errorAt(*this,
                       "Incompatible input shapes. Expected gatesInput shape: [batch_size, 4*hidden_size], "
                       "initialCellState shape: [batch_size, hidden_size]. Got gatesInput shape: {0}, initialCellState "
                       "shape: {1}",
                       gatesInputShape, initialCellStateShape);
    }

    return mlir::success();
}
