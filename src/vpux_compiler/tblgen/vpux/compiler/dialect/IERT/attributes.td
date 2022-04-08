//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_IERT_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_IERT_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/IERT/dialect.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"

//
// MemRefAttr
//

def IERT_MemRefAttr :
        StructAttr<
            "MemRefAttr",
            IERT_Dialect,
            [
                StructFieldAttr<"order", AffineMapAttr>,
                StructFieldAttr<"strides", I64ArrayAttr>,
            ]
        >;

#endif
