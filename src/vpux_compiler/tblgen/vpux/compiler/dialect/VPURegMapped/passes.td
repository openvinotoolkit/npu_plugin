//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_PASSES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_PASSES

include "mlir/Pass/PassBase.td"

def ResolveTaskLocations : PassBase<"resolve-mapped-inference-task-locations", "vpux::FunctionPass"> {
    let summary = [{ Resolve mapped inference Task operations metadata locations}];

    let constructor = "vpux::VPURegMapped::resolveMappedInferenceTaskLocationsPass()";
}

#endif
