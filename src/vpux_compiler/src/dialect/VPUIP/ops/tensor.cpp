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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// DeclareTensorOp
//

void vpux::VPUIP::DeclareTensorOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type memory,
                                         VPUIP::MemoryLocation locale, uint64_t dataIndex) {
    build(builder, state, memory, locale,
          nullptr,  // localeIndex
          dataIndex,
          nullptr,  // sparsityIndex
          nullptr,  // storageElementIndex
          nullptr,  // storageElementSize
          nullptr,  // leadingOffset
          nullptr   // trailingOffset
    );
}

void vpux::VPUIP::DeclareTensorOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type memory,
                                         VPUIP::MemoryLocation locale, uint32_t localeIndex, uint64_t dataIndex) {
    build(builder, state, memory, locale, getInt32Attr(builder.getContext(), localeIndex), dataIndex,
          nullptr,  // sparsityIndex
          nullptr,  // storageElementIndex
          nullptr,  // storageElementSize
          nullptr,  // leadingOffset
          nullptr   // trailingOffset
    );
}

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareTensorOp op) {
    const auto locale = op.locale();

    // TODO: check localeIndex

    const auto memref = op.memory().getType().cast<mlir::MemRefType>();

    if (!isMemoryCompatible(locale, memref)) {
        return errorAt(op, "Locale '{0}' is not compatible with memory space '{1}'", locale, memref.getMemorySpace());
    }

    // TODO: check other offsets

    return mlir::success();
}

//
// DeclareConstantTensorOp
//

void vpux::VPUIP::DeclareConstantTensorOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                 mlir::MemRefType type, mlir::ElementsAttr value) {
    build(builder, state, type, value, false);
}

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareConstantTensorOp op) {
    const auto memref = op.getType();
    const auto mem = getPhysicalMemory(memref);

    if (mlir::failed(mem) || mem.getValue() != VPUIP::PhysicalMemory::DDR) {
        return errorAt(op, "Unsupported result memory space '{0}'", memref.getMemorySpace());
    }

    return mlir::success();
}
