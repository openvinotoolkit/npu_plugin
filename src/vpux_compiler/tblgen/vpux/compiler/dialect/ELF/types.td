//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
// Symbol
//

def ELF_Symbol : ELF_Type<"Symbol"> {
    let summary = "ELF Symbol Type";

    let description = "This object represents closely a Symbol";
}



//
// Section
//

def ELF_Section : ELF_Type<"Section"> {
    let summary = "ELF Section Type";

    let description = "This object represents closely a Section";
}

#endif
