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

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>

namespace vpux {

//
// SimpleParser
//

struct SimpleParser final {
    //
    // High-level API
    //

    template <class ConcreteClass>
    static mlir::Attribute parseAttr(mlir::DialectAsmParser& parser) {
        return parseClass<mlir::Attribute, ConcreteClass>(parser, "Attribute");
    }

    template <class ConcreteClass>
    static mlir::Type parseType(mlir::DialectAsmParser& parser) {
        return parseClass<mlir::Type, ConcreteClass>(parser, "Type");
    }

    //
    // Low-level API
    //

    static mlir::LogicalResult parseValue(mlir::DialectAsmParser& parser, StringRef baseClass, StringRef mnemonic,
                                          FuncRef<mlir::LogicalResult()> valueParser);

    template <class BaseClass>
    static BaseClass parseClass(mlir::DialectAsmParser& parser, StringRef baseClass, StringRef mnemonic,
                                FuncRef<mlir::LogicalResult()> valueParser, FuncRef<BaseClass()> creator) {
        if (mlir::failed(parseValue(parser, baseClass, mnemonic, valueParser))) {
            return nullptr;
        }

        return creator();
    }

    template <class BaseClass, class ConcreteClass>
    static BaseClass parseClass(mlir::DialectAsmParser& parser, StringRef baseClass) {
        using ValueType = typename ConcreteClass::ValueType;

        ValueType val = {};

        return parseClass<BaseClass>(
                parser, baseClass, ConcreteClass::getMnemonic(),
                [&parser, &val]() -> mlir::LogicalResult {
                    return ConcreteClass::parseValue(parser, val);
                },
                [&parser, &val]() -> BaseClass {
                    return ConcreteClass::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()), val);
                });
    }
};

//
// ArrayParser
//

struct ArrayParser final {
    //
    // High-level API
    //

    template <class ConcreteClass>
    static mlir::Attribute parseAttr(mlir::DialectAsmParser& parser) {
        return parseClass<mlir::Attribute, ConcreteClass>(parser, "Attribute");
    }

    template <class ConcreteClass>
    static mlir::Type parseType(mlir::DialectAsmParser& parser) {
        return parseClass<mlir::Type, ConcreteClass>(parser, "Type");
    }

    //
    // Low-level API
    //

    static mlir::LogicalResult parseArray(mlir::DialectAsmParser& parser, FuncRef<mlir::LogicalResult()> itemParser);

    template <class BaseClass, class ConcreteClass>
    static BaseClass parseClass(mlir::DialectAsmParser& parser, StringRef baseClass) {
        using ItemType = typename ConcreteClass::ValueType::value_type;

        SmallVector<ItemType, 4> vec;

        return SimpleParser::parseClass<BaseClass>(
                parser, baseClass, ConcreteClass::getMnemonic(),
                [&parser, &vec]() -> mlir::LogicalResult {
                    return parseArray(parser, [&parser, &vec]() -> mlir::LogicalResult {
                        ItemType val = {};
                        if (mlir::failed(ConcreteClass::parseItem(parser, val))) {
                            return mlir::failure();
                        }

                        vec.push_back(std::move(val));
                    });
                },
                [&parser, &vec]() -> BaseClass {
                    return ConcreteClass::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
                                                     makeArrayRef(vec));
                });
    }
};

//
// TupleParser
//

struct TupleParser final {
    //
    // High-level API
    //

    template <class ConcreteClass>
    static mlir::Attribute parseAttr(mlir::DialectAsmParser& parser) {
        return parseClass<mlir::Attribute, ConcreteClass>(parser, "Attribute");
    }

    template <class ConcreteClass>
    static mlir::Type parseType(mlir::DialectAsmParser& parser) {
        return parseClass<mlir::Type, ConcreteClass>(parser, "Type");
    }

    //
    // Low-level API
    //

    static mlir::LogicalResult parseTuple(mlir::DialectAsmParser& parser,
                                          FuncRef<mlir::LogicalResult()> tupleItemsParser);

    template <size_t Index, class ConcreteClass, class TupleType>
    static auto parseTupleItems(mlir::DialectAsmParser& parser, StringRef, TupleType& tuple)
            -> std::enable_if_t<Index + 1 == std::tuple_size<TupleType>::value, mlir::LogicalResult> {
        return ConcreteClass::template parseItem<Index>(parser, std::get<Index>(tuple));
    }

    template <size_t Index, class ConcreteClass, class TupleType>
    static auto parseTupleItems(mlir::DialectAsmParser& parser, StringRef baseClass, TupleType& tuple)
            -> std::enable_if_t<Index + 1 < std::tuple_size<TupleType>::value, mlir::LogicalResult> {
        if (mlir::failed(ConcreteClass::template parseItem<Index>(parser, std::get<Index>(tuple)))) {
            return mlir::failure();
        }

        if (mlir::failed(parser.parseComma())) {
            return printTo(parser.emitError(parser.getCurrentLocation()),
                           "{0} {1} inner values should be divided with ',' symbol", ConcreteClass::getMnemonic(),
                           baseClass);
        }

        return parseTupleItems<Index + 1, ConcreteClass>(parser, baseClass, tuple);
    }

    template <class BaseClass, class ConcreteClass>
    static BaseClass parseClass(mlir::DialectAsmParser& parser, StringRef baseClass) {
        using ValueType = typename ConcreteClass::ValueType;

        ValueType tuple = {};

        return SimpleParser::parseClass<BaseClass>(
                parser, baseClass, ConcreteClass::getMnemonic(),
                [&parser, &tuple, baseClass]() -> mlir::LogicalResult {
                    return parseTuple(parser, [&parser, &tuple, baseClass]() -> mlir::LogicalResult {
                        return parseTupleItems<0, ConcreteClass>(parser, baseClass, tuple);
                    });
                },
                [&parser, &tuple]() -> BaseClass {
                    return ConcreteClass::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
                                                     std::move(tuple));
                });
    }

    //
    // Item parsers
    //

    template <typename T>
    static auto parseValue(mlir::DialectAsmParser& parser, T& val)
            -> enable_t<mlir::LogicalResult, std::is_integral<T>> {
        return parser.parseInteger(val);
    }

    template <typename T>
    static auto parseValue(mlir::DialectAsmParser& parser, T& val)
            -> enable_t<mlir::LogicalResult, std::is_floating_point<T>> {
        double temp = 0;
        if (mlir::failed(parser.parseFloat(temp))) {
            return mlir::failure();
        }
        val = checked_cast<T>(temp);
        return mlir::success();
    }

    template <typename T>
    static mlir::LogicalResult parseValue(mlir::DialectAsmParser& parser, std::vector<T>& arr) {
        return ArrayParser::parseArray(parser, [&parser, &arr]() -> mlir::LogicalResult {
            T val = 0;

            if (mlir::failed(parseValue(parser, val))) {
                return mlir::failure();
            }

            arr.push_back(val);
            return mlir::success();
        });
    }

    template <class Attr>
    static auto parseValue(mlir::DialectAsmParser& parser, Attr& attr)
            -> enable_t<mlir::LogicalResult, std::is_base_of<mlir::Attribute, Attr>> {
        return parser.parseAttribute(attr);
    }

    template <class Type>
    static auto parseValue(mlir::DialectAsmParser& parser, Type& type)
            -> enable_t<mlir::LogicalResult, std::is_base_of<mlir::Type, Type>> {
        return parser.parseType(type);
    }

    static mlir::LogicalResult parseItem(mlir::DialectAsmParser& parser, StringRef mnemonic, StringRef itemName,
                                         FuncRef<mlir::LogicalResult()> itemParser);

    template <typename T>
    static mlir::LogicalResult parseItem(mlir::DialectAsmParser& parser, StringRef mnemonic, StringRef itemName,
                                         T& item) {
        return parseItem(parser, mnemonic, itemName, [&parser, &item]() -> mlir::LogicalResult {
            return parseValue(parser, item);
        });
    }
};

}  // namespace vpux
