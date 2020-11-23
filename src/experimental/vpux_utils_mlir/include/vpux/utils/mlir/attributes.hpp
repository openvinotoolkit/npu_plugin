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

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/mlir/parser.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/StandardTypes.h>

namespace vpux {

//
// get*Type
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
// get*Attr
//

mlir::IntegerAttr getInt32Attr(mlir::MLIRContext* ctx, uint32_t val);
mlir::IntegerAttr getInt64Attr(mlir::MLIRContext* ctx, uint64_t val);

mlir::IntegerAttr getSInt32Attr(mlir::MLIRContext* ctx, int32_t val);
mlir::IntegerAttr getSInt64Attr(mlir::MLIRContext* ctx, int64_t val);

mlir::IntegerAttr getUInt32Attr(mlir::MLIRContext* ctx, uint32_t val);
mlir::IntegerAttr getUInt64Attr(mlir::MLIRContext* ctx, uint64_t val);

mlir::FloatAttr getFP32Attr(mlir::MLIRContext* ctx, float val);

//
// IntEnumAttr
//

template <typename Enum>
class IntEnumAttr : public mlir::IntegerAttr {
public:
    using ValueType = Enum;

public:
    using mlir::IntegerAttr::IntegerAttr;

public:
    static bool classof(mlir::Attribute attr) {
        return attr.isa<mlir::IntegerAttr>() && attr.getType().isSignlessInteger() &&
               EnumTraits<Enum>::isValidVal(attr.cast<mlir::IntegerAttr>().getInt());
    }

public:
    static auto get(mlir::MLIRContext* ctx, Enum val) {
        return getInt32Attr(ctx, static_cast<uint32_t>(val)).cast<IntEnumAttr>();
    }

public:
    auto getValue() const {
        return static_cast<Enum>(mlir::IntegerAttr::getInt());
    }
};

//
// StrEnumAttr
//

template <typename Enum>
class StrEnumAttr : public mlir::StringAttr {
public:
    using ValueType = Enum;

public:
    using mlir::StringAttr::StringAttr;

public:
    static bool classof(mlir::Attribute attr) {
        return attr.isa<mlir::StringAttr>() &&
               EnumTraits<Enum>::parseValue(attr.cast<mlir::StringAttr>().getValue()).hasValue();
    }

public:
    static auto get(mlir::MLIRContext* ctx, Enum val) {
        return mlir::StringAttr::get(EnumTraits<Enum>::getEnumValueName(val), ctx).template cast<StrEnumAttr>();
    }

public:
    auto getValue() const {
        return EnumTraits<Enum>::parseValue(mlir::StringAttr::getValue()).getValue();
    }
};

}  // namespace vpux
