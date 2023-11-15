//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/tensor_attr.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

constexpr StringLiteral orderName = "order";
constexpr StringLiteral memSpaceName = "mem_space";

bool vpux::TensorAttr::classof(mlir::Attribute attr) {
    if (attr == nullptr) {
        return false;
    }

    auto derived = attr.dyn_cast<mlir::DictionaryAttr>();
    if (derived == nullptr) {
        return false;
    }

    int numAbsentAttrs = 0;

    auto order = derived.get(orderName);
    if (order == nullptr) {
        ++numAbsentAttrs;
    } else if (!order.isa<mlir::AffineMapAttr>()) {
        return false;
    }

    auto memSpace = derived.get(memSpaceName);
    if (memSpace == nullptr) {
        ++numAbsentAttrs;
    } else if (!memSpace.isa<vpux::IndexedSymbolAttr>()) {
        return false;
    }

    return (derived.size() + numAbsentAttrs) == 2;
}

TensorAttr vpux::TensorAttr::get(mlir::MLIRContext* context, mlir::AffineMapAttr order,
                                 vpux::IndexedSymbolAttr memSpace) {
    SmallVector<mlir::NamedAttribute> fields;

    if (order != nullptr) {
        auto orderId = mlir::StringAttr::get(context, orderName);
        fields.emplace_back(orderId, order);
    }

    if (memSpace != nullptr) {
        auto memSpaceId = mlir::StringAttr::get(context, memSpaceName);
        fields.emplace_back(memSpaceId, memSpace);
    }

    auto dict = mlir::DictionaryAttr::get(context, fields);
    return dict.dyn_cast<TensorAttr>();
}

mlir::AffineMapAttr TensorAttr::getOrder() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    auto order = derived.get(orderName);
    if (order == nullptr) {
        return nullptr;
    }
    VPUX_THROW_WHEN(!order.isa<mlir::AffineMapAttr>(), "incorrect order Attribute type found: {0}", order);
    return order.cast<mlir::AffineMapAttr>();
}

vpux::IndexedSymbolAttr TensorAttr::getMemSpace() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    auto memSpace = derived.get(memSpaceName);
    if (memSpace == nullptr) {
        return nullptr;
    }
    VPUX_THROW_WHEN(!memSpace.isa<vpux::IndexedSymbolAttr>(), "incorrect mem space Attribute type found: {0}",
                    memSpace);
    return memSpace.cast<vpux::IndexedSymbolAttr>();
}

//
// Helpers
//

TensorAttr vpux::getTensorAttr(mlir::AffineMapAttr order, IndexedSymbolAttr memSpace) {
    // Initially, tensors do not have an encoding attribute, which is equivalent to an empty TensorAttr.
    // But in fact, such tensors have a different type: `tensor<1x8x4x2xf16> != tensor<1x8x4x2xf16, {}>`.
    // So let's not use empty attributes to avoid ambiguous representation of the same type.
    if ((order == nullptr || order.getValue().isIdentity()) && memSpace == nullptr) {
        return nullptr;
    }

    auto* ctx = order != nullptr ? order.getContext() : memSpace.getContext();

    return TensorAttr::get(ctx, order, memSpace);
}

TensorAttr vpux::getTensorAttr(mlir::AffineMap order, IndexedSymbolAttr memSpace) {
    return vpux::getTensorAttr(mlir::AffineMapAttr::get(order), memSpace);
}

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace) {
    return vpux::getTensorAttr(order.toAffineMap(ctx), memSpace);
}

TensorAttr vpux::getTensorAttr(mlir::RankedTensorType type) {
    if (const auto encoding = type.getEncoding()) {
        const auto tensorAttr = encoding.dyn_cast<TensorAttr>();
        VPUX_THROW_UNLESS(tensorAttr != nullptr, "Unsupported tensor encoding attribute '{0}'", encoding);

        return tensorAttr;
    }

    return nullptr;
}

mlir::AffineMap vpux::getOrder(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        if (const auto orderAttr = desc.getOrder()) {
            return orderAttr.getValue();
        }
    }

    const auto numDims = checked_cast<uint32_t>(type.getRank());
    return mlir::AffineMap::getMinorIdentityMap(numDims, numDims, type.getContext());
}

IndexedSymbolAttr vpux::getMemorySpace(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        return desc.getMemSpace();
    }

    return nullptr;
}
