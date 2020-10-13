//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"

//
// Base classes
//

class VPUIP_TypeDef<string name> : TypeDef<VPUIP_Dialect, name> {
    let mnemonic = name;
}

class VPUIP_Type<Pred pred, string descr> :
        DialectType<
            VPUIP_Dialect,
            pred,
            descr
        >;

//
// Barrier
//

def VPUIP_BarrierType : VPUIP_TypeDef<"Barrier"> {
    let summary = "This object represents closely a Barrier in the device";
}

def VPUIP_IsBarrier :
    CPred<
        "$_self.isa<vpux::VPUIP::BarrierType>()"
    >;

def VPUIP_Barrier :
        VPUIP_Type<
            VPUIP_IsBarrier,
            "VPUIP Barrier"
        >,
        BuildableType<
            "vpux::VPUIP::BarrierType::get($_builder.getContext())"
        > {
    let typeDescription = [{
        This object represents closely a Barrier in the device.
    }];
}

#endif
