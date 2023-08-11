//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IERT_TYPES
#define VPUX_COMPILER_DIALECT_IERT_TYPES

include "vpux/compiler/dialect/IERT/dialect.td"

include "mlir/IR/AttrTypeBase.td"

//
// Base classes
//

class IERT_Type<string name> : TypeDef<IERT_Dialect, name> {
    let mnemonic = name;
}

//
// IERT Packed Params
//

def IERT_PackedParams : IERT_Type<"PackedParams"> {
    let summary = "IERT Packed Params";

    let description = "IERT Packed Params";
}

#endif
