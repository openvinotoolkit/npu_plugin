//
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

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareTensorOp op) {
    const auto locale = op.locale();

    if (locale == MemoryLocation::ProgrammableInput || locale == MemoryLocation::ProgrammableOutput ||
        locale == MemoryLocation::GraphFile) {
        return errorAt(op, "MemoryLocation '{0}' can't be used for temporary tensor", locale);
    }

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
