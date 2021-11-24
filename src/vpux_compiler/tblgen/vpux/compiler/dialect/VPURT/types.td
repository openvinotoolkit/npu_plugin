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

#ifndef VPUX_COMPILER_DIALECT_VPURT_TYPES
#define VPUX_COMPILER_DIALECT_VPURT_TYPES

include "vpux/compiler/dialect/VPURT/dialect.td"

//
// Base classes
//

class VPURT_Type<string name> : TypeDef<VPURT_Dialect, name> {
    let mnemonic = name;
}

//
// Barrier
//

def VPURT_Barrier : VPURT_Type<"Barrier"> {
    let summary = "VPUIP Barrier Type";

    let description = "This object represents closely a Barrier in the device";
}

#endif