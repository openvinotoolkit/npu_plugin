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

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Types.h>

using namespace vpux;

//
// PostOp
//

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
// TensorAttr
//

IE::TensorAttr vpux::IE::getTensorAttr(mlir::AffineMapAttr order) {
    // Initially, tensors do not have an encoding attribute, which is equivalent to an empty TensorAttr.
    // But in fact, such tensors have a different type: `tensor<1x8x4x2xf16> != tensor<1x8x4x2xf16, {}>`.
    // So let's not use empty attributes to avoid ambiguous representation of the same type.
    if (order == nullptr || order.getValue().isIdentity()) {
        return nullptr;
    }

    return IE::TensorAttr::get(order, order.getContext());
}

IE::TensorAttr vpux::IE::getTensorAttr(mlir::AffineMap order) {
    return IE::getTensorAttr(mlir::AffineMapAttr::get(order));
}

IE::TensorAttr vpux::IE::getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order) {
    return IE::getTensorAttr(order.toPermutationAffineMap(ctx));
}

IE::TensorAttr vpux::IE::getTensorAttr(mlir::RankedTensorType origType) {
    if (const auto encoding = origType.getEncoding()) {
        const auto tensorAttr = encoding.dyn_cast<IE::TensorAttr>();
        VPUX_THROW_UNLESS(tensorAttr != nullptr, "Unsupported tensor encoding attribute '{0}'", encoding);

        return tensorAttr;
    }

    return nullptr;
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/structs.cpp.inc>
