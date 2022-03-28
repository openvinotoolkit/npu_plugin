//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section), /*sectionIndex=*/nullptr,
          getIntAttr(builder, byteOffset));
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
                                         int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, sectionIndex), getIntAttr(builder, byteOffset));
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, makeArrayRef({sectionIndex})), getIntAttr(builder, byteOffset));
}

mlir::LogicalResult vpux::VPURT::verifyOp(DeclareBufferOp op) {
    const auto type = op.getType().cast<vpux::NDTypeInterface>();
    const auto section = op.section();

    if (!VPURT::isMemoryCompatible(section, type)) {
        return errorAt(op, "BufferSection '{0}' is not compatible with memory space '{1}'", section,
                       type.getMemSpace());
    }

    // TODO: check sectionIndex and byteOffset [track: E#21111]

    return mlir::success();
}

SmallVector<int64_t> vpux::VPURT::DeclareBufferOp::getNonEmptySectionIndex() {
    if (sectionIndex().hasValue()) {
        return parseIntArrayAttr<int64_t>(sectionIndex().getValue());
    }
    return SmallVector<int64_t>({0});
}

//
// DeclareSparseBufferOp
//

mlir::ValueRange vpux::VPURT::DeclareSparseBufferOp::getViewSources() {
    return getOperands();
}

void vpux::VPURT::DeclareSparseBufferOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               mlir::Value data) {
    const auto inputType = data.getType().cast<mlir::MemRefType>();
    const auto sparseBuffer = VPURT::SparseBufferType::get(inputType);
    build(odsBuilder, odsState, sparseBuffer, data, nullptr, nullptr);
}

void vpux::VPURT::DeclareSparseBufferOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               mlir::Value data, mlir::Value sparsityMap) {
    const auto inputType = data.getType().cast<mlir::MemRefType>();
    const auto sparsityMapType = sparsityMap.getType().cast<mlir::MemRefType>();
    const auto sparseBuffer = VPURT::SparseBufferType::get(inputType, sparsityMapType);
    build(odsBuilder, odsState, sparseBuffer, data, sparsityMap, nullptr);
}

void vpux::VPURT::DeclareSparseBufferOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               mlir::Value data, mlir::Value sparsityMap,
                                               mlir::Value storageElementTable) {
    const auto inputType = data.getType().cast<mlir::MemRefType>();
    const auto sparsityMapType = sparsityMap.getType().cast<mlir::MemRefType>();
    const auto storageElementTableType = storageElementTable.getType().cast<mlir::MemRefType>();
    const auto sparseBuffer = VPURT::SparseBufferType::get(inputType, sparsityMapType, storageElementTableType);
    build(odsBuilder, odsState, sparseBuffer, data, sparsityMap, storageElementTable);
}
