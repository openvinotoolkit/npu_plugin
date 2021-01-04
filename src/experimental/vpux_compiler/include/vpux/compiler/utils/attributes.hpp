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
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getInt32Attr(ctx, checked_cast<uint32_t>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getInt64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getInt64Attr(ctx, checked_cast<uint64_t>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getSInt32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getSInt32Attr(ctx, checked_cast<int32_t>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getSInt64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getSInt64Attr(ctx, checked_cast<int64_t>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getUInt32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getUInt32Attr(ctx, checked_cast<uint32_t>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getUInt64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getUInt64Attr(ctx, checked_cast<uint64_t>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getFP32ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getFP32Attr(ctx, checked_cast<float>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

template <class Range>
mlir::ArrayAttr getFP64ArrayAttr(mlir::MLIRContext* ctx, Range range) {
    SmallVector<mlir::Attribute, 4> attrs;

    for (const auto val : range) {
        attrs.push_back(getFP64Attr(ctx, checked_cast<double>(val)));
    }

    return mlir::ArrayAttr::get(attrs, ctx);
}

//
// parse<scalar>ArrayAttr
//

SmallVector<int64_t, 4> parseIntArrayAttr(mlir::ArrayAttr arr);
SmallVector<double, 4> parseFPArrayAttr(mlir::ArrayAttr arr);

}  // namespace vpux
