//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ArchKindVPUX37XX
//

def VPUIPDPU_ArchKindVPUX37XX : NativeOpTrait<"ArchKindVPUX37XX"> {
    let cppNamespace = "vpux::VPUIPDPU";
}

#endif
