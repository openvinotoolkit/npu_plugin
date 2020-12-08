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

#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

//
// get<scalar>Type
//

mlir::IntegerType vpux::getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(32, ctx);
}

mlir::IntegerType vpux::getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(64, ctx);
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
