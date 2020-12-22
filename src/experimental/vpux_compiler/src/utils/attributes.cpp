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

SmallVector<int64_t, 4> vpux::parseIntArrayAttr(mlir::ArrayAttr arr) {
    return to_vector<4>(arr.getValue() | transformed([](mlir::Attribute attr) {
                            const auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>();
                            VPUX_THROW_UNLESS(intAttr != nullptr, "Got non Integer Attribute '{0}' in Array", attr);
                            VPUX_THROW_UNLESS(intAttr.getType().isSignlessInteger(),
                                              "Integer Attribute '{0}' is not signless", intAttr);

                            return intAttr.getInt();
                        }));
}

SmallVector<double, 4> vpux::parseFPArrayAttr(mlir::ArrayAttr arr) {
    return to_vector<4>(arr.getValue() | transformed([](mlir::Attribute attr) {
                            const auto fpAttr = attr.dyn_cast_or_null<mlir::FloatAttr>();
                            VPUX_THROW_UNLESS(fpAttr != nullptr, "Got non fpAttr Attribute '{0}' in Array", attr);

                            return fpAttr.getValueAsDouble();
                        }));
}
