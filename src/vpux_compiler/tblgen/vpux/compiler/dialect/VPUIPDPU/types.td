//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_TYPES
#define VPUX_COMPILER_DIALECT_VPUIPDPU_TYPES

include "vpux/compiler/dialect/VPUIPDPU/dialect.td"

//
// Base classes
//

class VPUIPDPU_Type<string name> : TypeDef<VPUIPDPU_Dialect, name> {
    let mnemonic = name;
}


//
// Invariant
//

def VPUIPDPU_Invariant : VPUIPDPU_Type<"Invariant"> {
    let summary = "VPUIPDPU Invariant Type";

    let description = "This object represents closely a DPU Invariant";
}

#endif
