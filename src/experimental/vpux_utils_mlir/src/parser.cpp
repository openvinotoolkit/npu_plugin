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

#include "vpux/utils/mlir/parser.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

//
// SimpleParser
//

mlir::LogicalResult vpux::SimpleParser::parseValue(
        mlir::DialectAsmParser& parser,
        StringRef baseClass,
        StringRef mnemonic,
        FuncRef<mlir::LogicalResult()> valueParser) {
    if (mlir::failed(parser.parseColon())) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "{0} should be followed by ':' symbol",
                       baseClass);
    }

    if (mlir::failed(valueParser())) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "Failed to parse inner part for {0} {1}",
                       mnemonic,
                       baseClass);
    }

    return mlir::success();
}

//
// ArrayParser
//

mlir::LogicalResult vpux::ArrayParser::parseArray(
        mlir::DialectAsmParser& parser,
        FuncRef<mlir::LogicalResult()> itemParser) {
    if (mlir::failed(parser.parseLSquare())) {
        return parser.emitError(parser.getCurrentLocation(),
                                "Array should be started with '[' symbol");
    }

    if (mlir::succeeded(parser.parseOptionalRSquare())) {
        return mlir::success();
    }

    do {
        if (mlir::failed(itemParser())) {
            return parser.emitError(parser.getCurrentLocation(),
                                    "Failed to parse array item");
        }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (mlir::failed(parser.parseRSquare())) {
        return parser.emitError(parser.getCurrentLocation(),
                                "Array should be ended with ']' symbol");
    }

    return mlir::success();
}

//
// TupleParser
//

mlir::LogicalResult vpux::TupleParser::parseTuple(
        mlir::DialectAsmParser& parser,
        FuncRef<mlir::LogicalResult()> tupleItemsParser) {
    if (mlir::failed(parser.parseLess())) {
        return parser.emitError(parser.getCurrentLocation(),
                                "Tuple should be started with '<' symbol");
    }

    if (mlir::failed(tupleItemsParser())) {
        return mlir::failure();
    }

    if (mlir::failed(parser.parseGreater())) {
        return parser.emitError(parser.getCurrentLocation(),
                                "Tuple should be ended with '>' symbol");
    }

    return mlir::success();
}

mlir::LogicalResult vpux::TupleParser::parseItem(
        mlir::DialectAsmParser& parser,
        StringRef mnemonic,
        StringRef itemName,
        FuncRef<mlir::LogicalResult()> itemParser) {
    if (mlir::failed(parser.parseKeyword(itemName))) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "Missing '{0}' item for {1}",
                       itemName,
                       mnemonic);
    }

    if (mlir::failed(parser.parseColon())) {
        return printTo(parser.emitError(parser.getCurrentLocation()),
                       "{0} '{1}' item should be followed by ':' symbol",
                       mnemonic,
                       itemName);
    }

    return itemParser();
}
