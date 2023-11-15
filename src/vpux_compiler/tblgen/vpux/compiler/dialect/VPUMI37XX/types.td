//
// Copyright Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI37XX_TYPES
#define VPUX_COMPILER_DIALECT_VPUMI37XX_TYPES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUMI37XX/dialect.td"
include "vpux/compiler/dialect/VPUMI37XX/attributes.td"

class VPUMI37XX_Type<string name, list<Trait> traits = [],
                          string baseCppClass = "::mlir::Type">
        : TypeDef<VPUMI37XX_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

#endif
