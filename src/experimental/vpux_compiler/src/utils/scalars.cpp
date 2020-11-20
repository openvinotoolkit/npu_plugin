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

#include "vpux/compiler/utils/scalars.hpp"

using namespace vpux;

//
// get<scalar>Type
//

mlir::IntegerType vpux::getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(32, mlir::IntegerType::Signless, ctx);
}

mlir::IntegerType vpux::getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(64, mlir::IntegerType::Signless, ctx);
}

mlir::IntegerType vpux::getSInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(8, mlir::IntegerType::Signed, ctx);
}

mlir::IntegerType vpux::getSInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(16, mlir::IntegerType::Signed, ctx);
}

mlir::IntegerType vpux::getSInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(32, mlir::IntegerType::Signed, ctx);
}

mlir::IntegerType vpux::getSInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(64, mlir::IntegerType::Signed, ctx);
}

mlir::IntegerType vpux::getUInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(8, mlir::IntegerType::Unsigned, ctx);
}

mlir::IntegerType vpux::getUInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(16, mlir::IntegerType::Unsigned, ctx);
}

mlir::IntegerType vpux::getUInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(32, mlir::IntegerType::Unsigned, ctx);
}

mlir::IntegerType vpux::getUInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(64, mlir::IntegerType::Unsigned, ctx);
}

//
// get<scalar>Attr
//

mlir::IntegerAttr vpux::getInt32Attr(mlir::MLIRContext* ctx, uint32_t val) {
    return mlir::IntegerAttr::get(getInt32Type(ctx), val);
}

mlir::IntegerAttr vpux::getInt64Attr(mlir::MLIRContext* ctx, uint64_t val) {
    return mlir::IntegerAttr::get(getInt64Type(ctx), val);
}

mlir::IntegerAttr vpux::getSInt32Attr(mlir::MLIRContext* ctx, int32_t val) {
    return mlir::IntegerAttr::get(getSInt32Type(ctx), val);
}

mlir::IntegerAttr vpux::getSInt64Attr(mlir::MLIRContext* ctx, int64_t val) {
    return mlir::IntegerAttr::get(getSInt64Type(ctx), val);
}

mlir::IntegerAttr vpux::getUInt32Attr(mlir::MLIRContext* ctx, uint32_t val) {
    return mlir::IntegerAttr::get(getUInt32Type(ctx), val);
}

mlir::IntegerAttr vpux::getUInt64Attr(mlir::MLIRContext* ctx, uint64_t val) {
    return mlir::IntegerAttr::get(getUInt64Type(ctx), val);
}

mlir::FloatAttr vpux::getFP32Attr(mlir::MLIRContext* ctx, float val) {
    return mlir::FloatAttr::get(mlir::FloatType::getF32(ctx), val);
}
