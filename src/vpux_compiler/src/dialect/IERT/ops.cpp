//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// initialize
//

void vpux::IERT::IERTDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IERT/generated/ops.cpp.inc>
            >();
}

//
// materializeConstant
//

mlir::Operation* vpux::IERT::IERTDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                              mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize IERT Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize IERT Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type.cast<vpux::NDTypeInterface>().eraseTiledInfo(),
                                            value.cast<Const::ContentAttr>());
}

//
// setupExtraInterfaces
//

void IERT::IERTDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addAttrInterface<mlir::BuiltinDialect, IERT::MemRefAttr, MemRefAttrLayout>();
}

//
// Operation executor attributes
//

namespace {

constexpr StringLiteral executorAttrName = "IERT.executor";

}  // namespace

void vpux::IERT::IERTDialect::setExecutor(mlir::async::ExecuteOp execOp, IndexedSymbolAttr executor) {
    VPUX_THROW_UNLESS(executor != nullptr, "Got an empty executor");
    execOp->setAttr(executorAttrName, executor);
}

llvm::StringLiteral vpux::IERT::IERTDialect::getExecutorAttrName() {
    return executorAttrName;
}

IndexedSymbolAttr vpux::IERT::IERTDialect::getExecutor(mlir::async::ExecuteOp execOp) {
    const auto executor = execOp->getAttr(executorAttrName);
    VPUX_THROW_UNLESS(executor != nullptr, "Can't find Executor attributes for Operation at '{0}'", execOp->getLoc());

    const auto executorSymbol = executor.dyn_cast<IndexedSymbolAttr>();
    VPUX_THROW_UNLESS(executorSymbol != nullptr, "Unsupported Executor attribute '{0}'", executorSymbol);

    return executorSymbol;
}

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IERT/generated/ops.cpp.inc>
