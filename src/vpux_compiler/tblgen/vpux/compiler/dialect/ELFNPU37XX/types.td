//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELFNPU37XX_TYPES
#define VPUX_COMPILER_DIALECT_ELFNPU37XX_TYPES

include "vpux/compiler/dialect/ELFNPU37XX/dialect.td"

include "mlir/IR/AttrTypeBase.td"

//
// Base classes
//

class ELFNPU37XX_Type<string name> : TypeDef<ELFNPU37XX_Dialect, name> {
    let mnemonic = name;
}

//
// Symbol
//

def ELFNPU37XX_Symbol : ELFNPU37XX_Type<"Symbol"> {
    let summary = "ELF Symbol Type";

    let description = "This object represents closely a Symbol";
}



//
// Section
//

def ELFNPU37XX_Section : ELFNPU37XX_Type<"Section"> {
    let summary = "ELF Section Type";

    let description = "This object represents closely a Section";
}

#endif
