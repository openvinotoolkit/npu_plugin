//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace edsl {

mlir::SmallVector<uint32_t, 4> padShapeTo4Dim(mlir::ArrayRef<int64_t> from);

template <typename Type>
mlir::SmallVector<Type, 4> getVectorFromArrayAttr(mlir::ArrayAttr attrs) {
    mlir::SmallVector<Type, 4> result;
    for (auto elem : attrs.getValue()) {
        result.emplace_back(elem.cast<mlir::IntegerAttr>().getInt());
    }
    return result;
}

MVCNN::DataType getSchemaDataType(mlir::Type type);

MVCNN::InitValue convertInitValue(mlir::Attribute attr);

}  // namespace edsl
}  // namespace vpux
