//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUMI37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// BarrierCompuation
//

def BarrierComputation : PassBase<"barrier-computation-VPUMI37XX", "vpux::FunctionPass"> {
    let summary = "Compute and setup scheduling related fields";

    let description = [{
        This pass computes and sets up the (barrier) scheduling related fields - next_same_id, start_after, clean_after.
        This logic was clasically part of the parser, but in ELF pipeline, it is included in the lowering.
    }];

    let constructor = "vpux::VPUMI37XX::createBarrierComputationPass()";

}

#endif
