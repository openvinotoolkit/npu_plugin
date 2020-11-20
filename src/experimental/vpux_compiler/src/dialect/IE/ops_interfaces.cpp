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

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

using namespace vpux;

//
// NetworkInformation
//

mlir::LogicalResult vpux::IE::verifyNetworkInformation(mlir::Operation* op) {
    if (!op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
        return printTo(op->emitError(),
                       "NetworkInformation Operation '{0}' is not Isolated",
                       *op);
    }

    if (!op->hasTrait<mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl>()) {
        return printTo(
                op->emitError(),
                "NetworkInformation Operation '{0}' is not attached to Module",
                *op);
    }

    if (op->getRegions().size() != 2) {
        return printTo(op->emitError(),
                       "NetworkInformation Operation '{0}' must have 2 Regions "
                       "attached with inputs/outputs information",
                       *op);
    }

    if (!op->hasTrait<mlir::OpTrait::NoRegionArguments>()) {
        return printTo(
                op->emitError(),
                "NetworkInformation Operation '{0}' Regions must have no "
                "arguments",
                *op);
    }

    if (mlir::dyn_cast<mlir::SymbolUserOpInterface>(op) == nullptr) {
        return printTo(
                op->emitError(),
                "NetworkInformation Operation '{0}' is not a Symbol User",
                *op);
    }

    const auto entryPointAttrName =
            mlir::Identifier::get("entryPoint", op->getContext());
    const auto entryPointAttr = op->getAttr(entryPointAttrName);
    if (entryPointAttr == nullptr) {
        return printTo(op->emitError(),
                       "NetworkInformation Operation '{0}' doesn't have '{1}' "
                       "attribute",
                       *op,
                       entryPointAttrName);
    }
    if (!entryPointAttr.isa<mlir::FlatSymbolRefAttr>()) {
        return printTo(op->emitError(),
                       "NetworkInformation Operation '{0}' attribute '{1}' is "
                       "not a Symbol Reference",
                       *op,
                       entryPointAttrName);
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/ops_interfaces.cpp.inc>
