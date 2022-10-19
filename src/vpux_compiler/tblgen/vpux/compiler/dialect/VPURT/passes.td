//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPURT_PASSES
#define VPUX_COMPILER_DIALECT_VPURT_PASSES

include "mlir/Pass/PassBase.td"

//
// AssignPhysicalBarriers
//

def AssignPhysicalBarriers : PassBase<"assign-physical-barriers", "vpux::FunctionPass"> {
    let summary = "Assign physical barriers";

    let description = [{
        This pass replaces virtual barriers with physical barriers and assign IDs to them.
    }];

    let constructor = "vpux::VPURT::createAssignPhysicalBarriersPass()";

    let options = [
        Option<
            "numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >
    ];
}

//
// AssignVirtualBarriers
//

def AssignVirtualBarriers : PassBase<"assign-virtual-barriers", "vpux::FunctionPass"> {
    let summary = "Schedules virtual barriers";

    let description = [{
        This pass inserts virtual barriers in the IR such that the number of active barriers at any time
        does not exceed the physical number of available barriers 
        and that the number of producers to a barrier is less than 256.
    }];

    let constructor = "vpux::VPURT::createAssignVirtualBarriersPass()";

    let options = [
        Option<
            "useCycleBasedBarrierScheduler", "use-cycle-based-barrier-scheduler",
            "bool", "true",
            "Flag to use cycle based barrier scheduler"
        >,
        Option<
            "numBarriers", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "numSlotsPerBarrier", "num-slots-per-barrier",
            "int", "",
            "Number of producer slots per barrier, available for use"
        >
    ];
}

//
// BarrierSimulation
//

def BarrierSimulation : PassBase<"barrier-simulation", "vpux::FunctionPass"> {
    let summary = "Simulate barriers";

    let description = [{
        Simulates the physical barrier execution and stops compilation on fail.
    }];

    let constructor = "vpux::VPURT::createBarrierSimulationPass()";
}

#endif
