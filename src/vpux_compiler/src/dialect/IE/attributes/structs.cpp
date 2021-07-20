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

#include "vpux/compiler/dialect/IE/attributes/structs.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Types.h>

using namespace vpux;

IE::PostOp vpux::IE::getPostOpAttr(mlir::MLIRContext* ctx, IE::PostOpKindAttr kindAttr,
                                   mlir::ArrayRef<mlir::NamedAttribute> attrs) {
    return IE::PostOp::get(kindAttr, mlir::DictionaryAttr::get(ctx, attrs), ctx);
}

IE::PostOpKind vpux::IE::getPostOpKind(IE::PostOp postOp) {
    return postOp.kind().getValue();
}

mlir::Attribute vpux::IE::getPostOpParam(IE::PostOp postOp, mlir::Identifier name) {
    return postOp.params().get(name);
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/structs.cpp.inc>
