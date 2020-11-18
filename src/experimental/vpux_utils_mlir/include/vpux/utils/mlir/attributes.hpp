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
// EnumAttr
//

namespace details {

template <typename Enum>
class EnumAttrStorage final : public mlir::AttributeStorage {
public:
    using KeyTy = Enum;

public:
    static auto construct(mlir::AttributeStorageAllocator& allocator,
                          Enum key) {
        const auto storage =
                new (allocator.allocate<EnumAttrStorage>()) EnumAttrStorage;

        storage->_key = key;

        return storage;
    }

public:
    auto getValue() const {
        return _key;
    }

public:
    bool operator==(Enum key) const {
        return _key == key;
    }

    static llvm::hash_code hashKey(Enum key) {
        return getHash(key);
    }

private:
    KeyTy _key = {};
};

}  // namespace details

template <class ConcreateAttr, typename Enum>
class EnumAttrBase
        : public mlir::Attribute::AttrBase<ConcreateAttr,
                                           mlir::Attribute,
                                           details::EnumAttrStorage<Enum>> {
public:
    using ValueType = Enum;

public:
    using EnumAttrBase::Base::Base;

    static ConcreateAttr get(mlir::MLIRContext* ctx, Enum val) {
        return EnumAttrBase::Base::get(ctx, val);
    }

    static ConcreateAttr getChecked(mlir::Location loc, Enum val) {
        return EnumAttrBase::Base::getChecked(loc, val);
    }

public:
    Enum getValue() const {
        return this->getImpl()->getValue();
    }

    operator Enum() const {
        return getValue();
    }

public:
    static mlir::Attribute parse(mlir::DialectAsmParser& parser) {
        return SimpleParser::parseAttr<ConcreateAttr>(parser);
    }

    void print(mlir::DialectAsmPrinter& os) const {
        printTo(os, "{0}:{1}", ConcreateAttr::getMnemonic(), getValue());
    }

private:
    friend SimpleParser;

    static mlir::LogicalResult parseValue(mlir::DialectAsmParser& parser,
                                          Enum& val) {
        StringRef keyword;
        if (mlir::failed(parser.parseKeyword(&keyword))) {
            return mlir::failure();
        }

        const auto parsed = EnumTraits<Enum>::parseValue(keyword);
        if (!parsed.hasValue()) {
            return printTo(parser.emitError(parser.getCurrentLocation()),
                           "Got invalid '{0}' case value : '{1}'",
                           ConcreateAttr::getMnemonic(),
                           keyword);
        }

        val = parsed.getValue();
        return mlir::success();
    }
};

}  // namespace vpux
