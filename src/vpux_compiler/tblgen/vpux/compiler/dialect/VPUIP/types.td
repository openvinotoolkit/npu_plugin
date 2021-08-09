//
// Copyright 2020 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"

//
// Base types
//

def I4  : I<4>;
def SI4  : SI<4>;
def UI4  : UI<4>;

//
// Base classes
//

class VPUIP_Type<string name> : TypeDef<VPUIP_Dialect, name> {
    let mnemonic = name;
}

//
// Barrier
//

def VPUIP_Barrier : VPUIP_Type<"Barrier"> {
    let summary = "VPUIP Barrier Type";

    let description = "This object represents closely a Barrier in the device";
}

#endif
