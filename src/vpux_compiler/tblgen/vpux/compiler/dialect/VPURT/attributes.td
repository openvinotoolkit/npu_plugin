//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPURT_ATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//
// BufferSection
//

def VPURT_BufferSection :
        StrEnumAttr<
            "BufferSection",
            "Values indicating which section of BLOB the buffer resides in",
            [
                StrEnumAttrCase<"NetworkInput">,
                StrEnumAttrCase<"NetworkOutput">,
                StrEnumAttrCase<"ProfilingOutput">,
                StrEnumAttrCase<"Constant">,
                StrEnumAttrCase<"SW_KernelText">,
                StrEnumAttrCase<"DDR">,
                StrEnumAttrCase<"CSRAM">,
                StrEnumAttrCase<"CMX_UPA">,
                StrEnumAttrCase<"CMX_NN">,
                StrEnumAttrCase<"Register">,
                StrEnumAttrCase<"MAC_Accumulators">,
            ]
        > {
    let cppNamespace = "vpux::VPURT";
    let genSpecializedAttr = 1;
}

#endif
