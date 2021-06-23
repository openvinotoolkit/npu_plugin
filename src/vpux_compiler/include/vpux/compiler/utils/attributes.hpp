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

#pragma once

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {

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

//
// get<scalar>ArrayAttr
//

template <class Range>
mlir::ArrayAttr getInt32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getInt32Attr(ctx, checked_cast<int32_t>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getInt64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getInt64Attr(ctx, checked_cast<int64_t>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getSInt32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getSInt32Attr(ctx, checked_cast<int32_t>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getSInt64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getSInt64Attr(ctx, checked_cast<int64_t>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getUInt32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getUInt32Attr(ctx, checked_cast<uint32_t>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getUInt64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getUInt64Attr(ctx, checked_cast<uint64_t>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getFP32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getFP32Attr(ctx, checked_cast<float>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getFP64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(getFP64Attr(ctx, checked_cast<double>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

template <class Range>
mlir::ArrayAttr getBoolArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute> attrs;

    for (const auto val : range) {
        attrs.push_back(mlir::BoolAttr::get(ctx, checked_cast<bool>(val)));
    }

    return mlir::ArrayAttr::get(ctx, attrs);
}

//
// parse<scalar>ArrayAttr
//

SmallVector<int64_t> parseIntArrayAttr(mlir::ArrayAttr arr);
SmallVector<double> parseFPArrayAttr(mlir::ArrayAttr arr);

}  // namespace vpux
