//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"

#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/attributes/enums.hpp.inc>
#include <vpux/compiler/dialect/IE/generated/attributes/structs.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/IE/generated/attributes.hpp.inc>

namespace vpux {
namespace IE {

//
// TensorAttr
//

IE::TensorAttr getTensorAttr(mlir::AffineMapAttr order, IndexedSymbolAttr memSpace);
IE::TensorAttr getTensorAttr(mlir::AffineMap order, IndexedSymbolAttr memSpace);
IE::TensorAttr getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace);

IE::TensorAttr getTensorAttr(mlir::RankedTensorType type);

mlir::AffineMap getOrder(mlir::RankedTensorType type);
IndexedSymbolAttr getMemorySpace(mlir::RankedTensorType type);

}  // namespace IE
}  // namespace vpux
