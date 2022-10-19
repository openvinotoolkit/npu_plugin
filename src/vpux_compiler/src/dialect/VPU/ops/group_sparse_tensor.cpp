//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

//
// build
//

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data) {
    build(builder, state, data, nullptr, nullptr);
}

void vpux::VPU::GroupSparseTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                           mlir::Value sparsityMap) {
    build(builder, state, data, sparsityMap, nullptr);
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::GroupSparseTensorOp::inferReturnTypes(mlir::MLIRContext*, Optional<mlir::Location>,
                                                                     mlir::ValueRange operands,
                                                                     mlir::DictionaryAttr /*attrs*/,
                                                                     mlir::RegionRange /*ranges*/,
                                                                     SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto data = operands[0];
    const auto sparsityMap = operands.size() > 1 ? operands[1] : nullptr;
    const auto storageElementTable = operands.size() > 2 ? operands[2] : nullptr;

    const auto dataType = data.getType().cast<mlir::TensorType>();
    const auto sparsityMapType = sparsityMap ? sparsityMap.getType().cast<mlir::TensorType>() : nullptr;
    const auto storageElementTableType =
            storageElementTable ? storageElementTable.getType().cast<mlir::TensorType>() : nullptr;

    inferredReturnTypes.push_back(VPU::SparseTensorType::get(dataType, sparsityMapType, storageElementTableType));

    return mlir::success();
}
