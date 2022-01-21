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
