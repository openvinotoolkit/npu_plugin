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

#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {

//
// get<Scalar>Attr
//

template <typename T>
mlir::IntegerAttr getIntAttr(mlir::MLIRContext* ctx, T val) {
    return mlir::IntegerAttr::get(getInt64Type(ctx), checked_cast<int64_t>(val));
}
template <typename T>
mlir::IntegerAttr getIntAttr(mlir::Builder& b, T val) {
    return getIntAttr(b.getContext(), val);
}

template <typename T>
mlir::FloatAttr getFPAttr(mlir::MLIRContext* ctx, T val) {
    return mlir::FloatAttr::get(mlir::FloatType::getF64(ctx), checked_cast<double>(val));
}
template <typename T>
mlir::FloatAttr getFPAttr(mlir::Builder& b, T val) {
    return getFPAttr(b.getContext(), val);
}

//
// get<Scalar>ArrayAttr
//

template <class Range>
mlir::ArrayAttr getIntArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getIntAttr(ctx, val));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}
template <class Range>
mlir::ArrayAttr getIntArrayAttr(mlir::Builder& b, Range range) {
    return getIntArrayAttr(b.getContext(), range);
}

template <class Range>
mlir::ArrayAttr getFPArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getFPAttr(ctx, val));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}
template <class Range>
mlir::ArrayAttr getFPArrayAttr(mlir::Builder& b, Range range) {
    return getFPArrayAttr(b.getContext(), range);
}

//
// parse<Scalar>ArrayAttr
//

template <typename T>
SmallVector<T> parseIntArrayAttr(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>();
                               VPUX_THROW_UNLESS(intAttr != nullptr, "Got non Integer Attribute '{0}' in Array", attr);

                               return checked_cast<T>(intAttr.getValue().getSExtValue());
                           }));
}

template <typename T>
SmallVector<T> parseFPArrayAttr(mlir::ArrayAttr arr) {
    return to_small_vector(arr.getValue() | transformed([](mlir::Attribute attr) {
                               const auto fpAttr = attr.dyn_cast_or_null<mlir::FloatAttr>();
                               VPUX_THROW_UNLESS(fpAttr != nullptr, "Got non fpAttr Attribute '{0}' in Array", attr);

                               return checked_cast<T>(fpAttr.getValueAsDouble());
                           }));
}

}  // namespace vpux
