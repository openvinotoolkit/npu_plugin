//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

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

mlir::FloatAttr vpux::getFP64Attr(mlir::MLIRContext* ctx, double val) {
    return mlir::FloatAttr::get(mlir::FloatType::getF64(ctx), val);
}

//
// parse<scalar>ArrayAttr
//

SmallVector<int64_t> vpux::parseIntArrayAttr(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>();
                               VPUX_THROW_UNLESS(intAttr != nullptr, "Got non Integer Attribute '{0}' in Array", attr);
                               VPUX_THROW_UNLESS(intAttr.getType().isSignlessInteger(),
                                                 "Integer Attribute '{0}' is not signless", intAttr);

                               return intAttr.getInt();
                           }));
}

SmallVector<double> vpux::parseFPArrayAttr(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto fpAttr = attr.dyn_cast_or_null<mlir::FloatAttr>();
                               VPUX_THROW_UNLESS(fpAttr != nullptr, "Got non fpAttr Attribute '{0}' in Array", attr);

                               return fpAttr.getValueAsDouble();
                           }));
}
