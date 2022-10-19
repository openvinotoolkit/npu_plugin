//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpImplementation.h>

namespace vpux {

//
// OptionalTypes
//

void printOptionalTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::TypeRange types);
mlir::ParseResult parseOptionalTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types);

template <typename... Args>
void printOptionalTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::Type type, Args... types) {
    printOptionalTypes(printer, op, mlir::TypeRange(makeArrayRef({type, types...})));
}

namespace details {

mlir::ParseResult parseOptionalTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> types);

}  // namespace details

template <typename... Args>
mlir::ParseResult parseOptionalTypes(mlir::OpAsmParser& parser, mlir::Type& type, Args&... types) {
    return details::parseOptionalTypes(parser, makeArrayRef({&type, &types...}));
}

//
// OptionalResultTypes
//

void printOptionalResultTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::TypeRange types);
mlir::ParseResult parseOptionalResultTypes(mlir::OpAsmParser& parser, SmallVectorImpl<mlir::Type>& types);

template <typename... Args>
void printOptionalResultTypes(mlir::OpAsmPrinter& printer, mlir::Operation* op, mlir::Type type, Args... types) {
    if (type != nullptr) {
        printOptionalResultTypes(printer, op, mlir::TypeRange(makeArrayRef({type, types...})));
    } else if (sizeof...(types) != 0) {
        SmallVector<mlir::Type> nonNullTypes;
        for (auto type : {types...}) {
            if (type != nullptr) {
                nonNullTypes.push_back(type);
            }
        }
        if (nonNullTypes.empty()) {
            return;
        }
        printOptionalResultTypes(printer, op, mlir::TypeRange(makeArrayRef(nonNullTypes)));
    }
}

namespace details {

mlir::ParseResult parseOptionalResultTypes(mlir::OpAsmParser& parser, ArrayRef<mlir::Type*> types);

}  // namespace details

template <typename... Args>
mlir::ParseResult parseOptionalResultTypes(mlir::OpAsmParser& parser, mlir::Type& type, Args&... types) {
    return details::parseOptionalResultTypes(parser, makeArrayRef({&type, &types...}));
}

//
// OptionalRegion
//

void printOptionalRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region);
mlir::ParseResult parseOptionalRegion(mlir::OpAsmParser& parser, mlir::Region& region);

//
// OptionalBlockRegion
//

void printOptionalBlockRegion(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::Region& region);
mlir::ParseResult parseOptionalBlockRegion(mlir::OpAsmParser& parser, mlir::Region& region);

}  // namespace vpux
