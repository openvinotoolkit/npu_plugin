//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/structs.hpp.inc>

namespace vpux {
namespace IE {

//
// TensorAttr
//

IE::TensorAttr getTensorAttr(mlir::AffineMapAttr order, IndexedSymbolAttr memSpace, bool sparse);
IE::TensorAttr getTensorAttr(mlir::AffineMap order, IndexedSymbolAttr memSpace, bool sparse);
IE::TensorAttr getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace, bool sparse);

IE::TensorAttr getTensorAttr(mlir::RankedTensorType type);

mlir::AffineMap getOrder(mlir::RankedTensorType type);
IndexedSymbolAttr getMemorySpace(mlir::RankedTensorType type);
bool isSparse(mlir::RankedTensorType type);

}  // namespace IE
}  // namespace vpux
