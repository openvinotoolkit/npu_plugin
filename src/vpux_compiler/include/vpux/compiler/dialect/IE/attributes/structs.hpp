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
