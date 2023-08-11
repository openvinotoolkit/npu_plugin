//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

static constexpr StringLiteral elidedLargeConstStr = "elided_large_const";

//
// OpaqueElementsAttr::print
//

void vpux::Const::OpaqueElementsAttr::print(mlir::AsmPrinter& printer) const {
    auto& flags = printer.getPrintingFlags();
    if (flags.shouldElideElementsAttr(*this)) {
        printer << printToString(R"(<"{0}", "0xDEADBEEF">)", elidedLargeConstStr);
    } else {
        printer << "<\"0x" << llvm::toHex(getValue()) << "\">";
    }
}

//
// OpaqueElementsAttr::parse
//

mlir::Attribute vpux::Const::OpaqueElementsAttr::parse(mlir::AsmParser& parser, mlir::Type type) {
    if (parser.parseLess()) {
        return nullptr;
    }

    std::string data;
    if (parser.parseString(&data)) {
        return nullptr;
    }

    if (elidedLargeConstStr.compare(data) == 0) {
        if (parser.parseComma()) {
            return nullptr;
        }

        if (parser.parseString(&data)) {
            return nullptr;
        }
    }

    if (data.size() < 2 || data.substr(0, 2) != "0x") {
        parser.emitError(parser.getNameLoc(), "Hex string should start with `0x`");
        return nullptr;
    }

    std::string hex;
    if (!llvm::tryGetFromHex(data.substr(2), hex)) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get string from hex: {0}", data.substr(2));
        return nullptr;
    }

    return OpaqueElementsAttr::get(type, hex);
}
