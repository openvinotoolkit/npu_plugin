//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/dialect.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
            >();

    registerTypes();
}

//
// Operation executor attributes
//

namespace {

constexpr StringLiteral executorAttrName = "VPUIP.executor";

}  // namespace

void vpux::VPUIP::VPUIPDialect::setExecutor(mlir::async::ExecuteOp execOp, IndexedSymbolAttr executor) {
    VPUX_THROW_UNLESS(executor != nullptr, "Got an empty executor");
    execOp->setAttr(executorAttrName, executor);
}

llvm::StringLiteral vpux::VPUIP::VPUIPDialect::getExecutorAttrName() {
    return executorAttrName;
}

IndexedSymbolAttr vpux::VPUIP::VPUIPDialect::getExecutor(mlir::async::ExecuteOp execOp) {
    const auto executor = execOp->getAttr(executorAttrName);
    VPUX_THROW_UNLESS(executor != nullptr, "Can't find Executor attributes for Operation at '{0}'", execOp->getLoc());

    const auto executorSymbol = executor.dyn_cast<IndexedSymbolAttr>();
    VPUX_THROW_UNLESS(executorSymbol != nullptr, "Unsupported Executor attribute '{0}'", executorSymbol);

    return executorSymbol;
}

//
// materializeConstant
//

mlir::Operation* vpux::VPUIP::VPUIPDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize VPUIP Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize VPUIP Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type.cast<vpux::NDTypeInterface>().eraseTiledInfo(),
                                            value.cast<Const::ContentAttr>());
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/dialect.cpp.inc>
