//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELFNPU37XX
#define VPUX_COMPILER_DIALECT_ELFNPU37XX

include "mlir/IR/OpBase.td"

// Provide a definition of the ELF dialect in the ODS framework
def ELFNPU37XX_Dialect : Dialect {
    let name = "ELFNPU37XX";

    let cppNamespace = "::vpux::ELFNPU37XX";

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

#endif
