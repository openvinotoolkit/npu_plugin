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

#ifndef VPUX_COMPILER_DIALECT_ELF_TYPES
#define VPUX_COMPILER_DIALECT_ELF_TYPES

include "vpux/compiler/dialect/ELF/dialect.td"

//
// Base classes
//

class ELF_Type<string name> : TypeDef<ELF_Dialect, name> {
    let mnemonic = name;
}

//
// Section
//

def ELF_Section : ELF_Type<"Section"> {
    let summary = "ELF Section Type";

    let description = "This object represents closely a Section";
}

#endif
