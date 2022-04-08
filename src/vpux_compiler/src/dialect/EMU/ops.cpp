//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TensorEncoding.h>
#include <mlir/Interfaces/DecodeAttributesInterfaces.h>

using namespace vpux;

//
// initialize
//

void EMU::EMUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/EMU/generated/ops.cpp.inc>
            >();
}

//
// materializeConstant
//

mlir::Operation* EMU::EMUDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type,
                                                      mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize EMU Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        (void)errorAt(loc, "Can't materialize EMU Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

#include <vpux/compiler/dialect/EMU/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/EMU/generated/ops.cpp.inc>
