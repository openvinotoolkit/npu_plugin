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

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

//
// get<scalar>Type
//

mlir::IntegerType getInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt64Type(mlir::MLIRContext* ctx);

mlir::IntegerType getSInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt64Type(mlir::MLIRContext* ctx);

mlir::IntegerType getUInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt64Type(mlir::MLIRContext* ctx);

//
// get<scalar>Attr
//

mlir::IntegerAttr getInt32Attr(mlir::MLIRContext* ctx, uint32_t val);
mlir::IntegerAttr getInt64Attr(mlir::MLIRContext* ctx, uint64_t val);

mlir::IntegerAttr getSInt32Attr(mlir::MLIRContext* ctx, int32_t val);
mlir::IntegerAttr getSInt64Attr(mlir::MLIRContext* ctx, int64_t val);

mlir::IntegerAttr getUInt32Attr(mlir::MLIRContext* ctx, uint32_t val);
mlir::IntegerAttr getUInt64Attr(mlir::MLIRContext* ctx, uint64_t val);

mlir::FloatAttr getFP32Attr(mlir::MLIRContext* ctx, float val);
mlir::FloatAttr getFP64Attr(mlir::MLIRContext* ctx, double val);

}  // namespace vpux
