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
