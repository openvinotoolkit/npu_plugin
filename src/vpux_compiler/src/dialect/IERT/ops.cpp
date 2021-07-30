//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>

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

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// Operation executor attributes
//

namespace {

constexpr StringLiteral executorAttrName = "IERT.executor";
constexpr StringLiteral numUnitsAttrName = "IERT.num_units";

}  // namespace

void vpux::IERT::IERTDialect::setExecutor(mlir::async::ExecuteOp execOp, mlir::Attribute executor, uint32_t numUnits) {
    execOp->setAttr(executorAttrName, executor);
    execOp->setAttr(numUnitsAttrName, getIntAttr(execOp->getContext(), numUnits));
}

mlir::Attribute vpux::IERT::IERTDialect::getExecutor(mlir::async::ExecuteOp execOp, uint32_t& numUnits) {
    if (const auto executor = execOp->getAttr(executorAttrName)) {
        const auto numUnitsAttr = execOp->getAttr(numUnitsAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
        VPUX_THROW_UNLESS(numUnitsAttr != nullptr,
                          "'{0}' attribute was not set, it must be used together with '{1}' attribute", numUnitsAttr,
                          executorAttrName);

        numUnits = checked_cast<uint32_t>(numUnitsAttr.getInt());
        return executor;
    }

    VPUX_THROW("Can't find Executor attributes for Operation at '{0}'", execOp->getLoc());
}

//
// Generated
//

#include <vpux/compiler/dialect/IERT/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IERT/generated/ops.cpp.inc>
