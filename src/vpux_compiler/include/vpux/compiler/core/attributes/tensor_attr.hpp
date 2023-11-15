//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"

namespace vpux {

//
// TensorAttr
//

class TensorAttr : public mlir::DictionaryAttr {
public:
    using mlir::DictionaryAttr::DictionaryAttr;

public:
    static bool classof(mlir::Attribute attr);

public:
    static TensorAttr get(mlir::MLIRContext* context, mlir::AffineMapAttr order, vpux::IndexedSymbolAttr memSpace);

public:
    mlir::AffineMapAttr getOrder() const;
    vpux::IndexedSymbolAttr getMemSpace() const;
};

//
// Helpers
//

TensorAttr getTensorAttr(mlir::AffineMapAttr order, vpux::IndexedSymbolAttr memSpace);
TensorAttr getTensorAttr(mlir::AffineMap order, vpux::IndexedSymbolAttr memSpace);
TensorAttr getTensorAttr(mlir::MLIRContext* ctx, vpux::DimsOrder order, vpux::IndexedSymbolAttr memSpace);
TensorAttr getTensorAttr(mlir::RankedTensorType type);

mlir::AffineMap getOrder(mlir::RankedTensorType type);
vpux::IndexedSymbolAttr getMemorySpace(mlir::RankedTensorType type);

}  // namespace vpux
