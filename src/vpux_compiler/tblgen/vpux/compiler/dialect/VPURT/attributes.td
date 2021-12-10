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

#ifndef VPUX_COMPILER_DIALECT_VPURT_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPURT_ATTRIBUTES

include "mlir/IR/OpBase.td"

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
