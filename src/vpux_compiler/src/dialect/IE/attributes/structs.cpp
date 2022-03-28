//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/attributes/structs.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Types.h>

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/structs.cpp.inc>

using namespace vpux;

//
// TensorAttr
//

IE::TensorAttr vpux::IE::getTensorAttr(mlir::AffineMapAttr order, IndexedSymbolAttr memSpace, bool sparse) {
    // Initially, tensors do not have an encoding attribute, which is equivalent to an empty TensorAttr.
    // But in fact, such tensors have a different type: `tensor<1x8x4x2xf16> != tensor<1x8x4x2xf16, {}>`.
    // So let's not use empty attributes to avoid ambiguous representation of the same type.
    if ((order == nullptr || order.getValue().isIdentity()) && memSpace == nullptr && !sparse) {
        return nullptr;
    }

    auto* ctx = order != nullptr ? order.getContext() : memSpace.getContext();
    auto sparseAttr = sparse ? mlir::UnitAttr::get(ctx) : nullptr;

    return IE::TensorAttr::get(order, memSpace, sparseAttr, ctx);
}

IE::TensorAttr vpux::IE::getTensorAttr(mlir::AffineMap order, IndexedSymbolAttr memSpace, bool sparse) {
    return IE::getTensorAttr(mlir::AffineMapAttr::get(order), memSpace, sparse);
}

IE::TensorAttr vpux::IE::getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace,
                                       bool sparse) {
    return IE::getTensorAttr(order.toAffineMap(ctx), memSpace, sparse);
}

IE::TensorAttr vpux::IE::getTensorAttr(mlir::RankedTensorType type) {
    if (const auto encoding = type.getEncoding()) {
        const auto tensorAttr = encoding.dyn_cast<IE::TensorAttr>();
        VPUX_THROW_UNLESS(tensorAttr != nullptr, "Unsupported tensor encoding attribute '{0}'", encoding);

        return tensorAttr;
    }

    return nullptr;
}

mlir::AffineMap vpux::IE::getOrder(mlir::RankedTensorType type) {
    if (const auto desc = IE::getTensorAttr(type)) {
        if (const auto orderAttr = desc.order()) {
            return orderAttr.getValue();
        }
    }

    const auto numDims = checked_cast<uint32_t>(type.getRank());
    return mlir::AffineMap::getMinorIdentityMap(numDims, numDims, type.getContext());
}

IndexedSymbolAttr vpux::IE::getMemorySpace(mlir::RankedTensorType type) {
    if (const auto desc = IE::getTensorAttr(type)) {
        return desc.mem_space();
    }

    return nullptr;
}

bool vpux::IE::isSparse(mlir::RankedTensorType type) {
    if (const auto desc = IE::getTensorAttr(type)) {
        return desc.sparse() != nullptr;
    }

    return false;
}
