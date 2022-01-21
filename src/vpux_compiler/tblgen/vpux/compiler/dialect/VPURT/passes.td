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
