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

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_TYPES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_TYPES

include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"

//
// Base classes
//

class VPUIPRegMapped_Type<string name> : TypeDef<VPUIPRegMapped_Dialect, name> {
    let mnemonic = name;
}

//
// Barrier
//

def VPUIPRegMapped_Barrier : VPUIPRegMapped_Type<"Barrier"> {
    let summary = "VPUIPRegMapped Barrier Type";

    let description = "This object represents closely a Barrier in the device";
}

#endif
