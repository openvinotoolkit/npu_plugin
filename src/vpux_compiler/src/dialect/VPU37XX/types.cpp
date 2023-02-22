//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU37XX/types.hpp"
#include "vpux/compiler/dialect/VPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

// define explicitly generatedTypeParser() and generatedTypePrinter()
// as none of the types of this dialect has 'let mnemonic = ' set.
// 'let mnemonic = ' is intentionally unset in order to avoid declaring Specialized_Type::get(ctx) method

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser& parser, ::llvm::StringRef mnemonic,
                                                       ::mlir::Type& value) {
    VPUX_UNUSED(parser);
    VPUX_UNUSED(mnemonic);
    VPUX_UNUSED(value);
    return {};
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter& printer) {
    VPUX_UNUSED(def);
    VPUX_UNUSED(printer);
    return ::mlir::success();
}

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU37XX/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// register Types
//

void vpux::VPU37XX::VPU37XXDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPU37XX/generated/types.cpp.inc>
#undef GET_TYPEDEF_LIST
            >();
}
