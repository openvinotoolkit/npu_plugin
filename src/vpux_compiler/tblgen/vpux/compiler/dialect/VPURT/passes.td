//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
// SimplifySchedule
//

def SimplifySchedule : PassBase<"simplify-schedule", "vpux::FunctionPass"> {
    let summary = "Simplify schedule for runtime";

    let description = [{
        This pass will simplify schedule by: 
            1. sharing wait / update barriers for DMAs
            2. removing cycled dependencies where possible
    }];

    let constructor = "vpux::VPURT::createSimplifySchedulePass()";
}

//
// SplitExceedingVariantCountBarriers
//

def SplitExceedingVariantCountBarriers : PassBase<"split-exceeding-variant-count-barriers", "vpux::FunctionPass"> {
    let summary = "Split exceeding variant (producer and consumer) count barriers";

    let description = [{
        This pass splits virtual barriers in the IR such that the number of producer slots <= MAX_VARIANT_COUNT / 2 
        and the number of consumer slots <= MAX_VARIANT_COUNT / 2.
        Besides, there's a more strict limit named MAX_VARIANT_SUM from hw runtime. The value may be smaller than MAX_VARIANT_COUNT on same Arch.
        This limit will be enabled when MAX_VARIANT_SUM < MAX_VARIANT_COUNT and then check producers + consumers <= MAX_VARIANT_SUM. If not satisfied,
        It will split barriers to keep producers <= MAX_VARIANT_SUM / 2 and consumers <= MAX_VARIANT_SUM / 2.
    }];

    let constructor = "vpux::VPURT::createSplitExceedingVariantCountBarriersPass()";

    let options = [
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "To control available number (a half of maxVariantCount) of producer and consumer slots per barrier"
        >,
        Option<
            "maxVariantSum", "max-variant-sum",
            "int", "",
            "Maximum sum of producer and consumer slots per barrier"
        >
    ];
}

//
// SatisfyOneWaitBarrierPerTask
//

def SatisfyOneWaitBarrierPerTask : PassBase<"satisfy-one-wait-barrier-per-task", "vpux::FunctionPass"> {
    let summary = "Satisfy each task has one wait barrier";

    let description = [{
        This pass will satisfy barrier safety constraint
        -- single barrier drives each task
        Handling multi barrier input can be wasteful and considerably more difficult to manage 
        barrier safety. Pass is manipulating tasks with parallel wait barriers by two modes:
        1. Merge parallel wait barriers with legal barrier variant count
        2. Linearize parallel wait barrier producers with exceeding variant count
    }];

    let constructor = "vpux::VPURT::createSatisfyOneWaitBarrierPerTaskPass()";

    let options = [
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "Number of producer and consumer slots per barrier, available for use"
        >
    ];
}

//
// ReduceExceedingActiveCountBarriers
//

def ReduceExceedingActiveCountBarriers : PassBase<"reduce-exceeding-active-count-barriers", "vpux::FunctionPass"> {
    let summary = "Reduce exceeding active barrier count";

    let description = [{
        This pass linearizes virtual barriers in the IR such that the number of active barriers at any time
        does not exceed the physical number of available barriers and that total producer + consumer variant 
        count <= MAX_VARIANT_COUNT.
        With exceeding active barrier count:
        - parallel barriers can be merged
        - parallel consumers to barrier can be linearized
        - parallel barriers can be linearized
        If nothing can be applied extreme linearization will be performed, where all tasks linked to active barriers
        will be linearized. 
    }];

    let constructor = "vpux::VPURT::createReduceExceedingActiveCountBarriersPass()";

    let options = [
        Option<
            "numBarriers", "num-barriers",
            "int", "",
            "Number of physical barriers, available for use"
        >,
        Option<
            "maxVariantCount", "max-variant-count",
            "int", "",
            "Number of producer and consumer slots per barrier, available for use"
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

//
// InferenceExecutionAnalysis
//

def InferenceExecutionAnalysis : PassBase<"inference-execution-analysis", "vpux::FunctionPass"> {
    let summary = "Perform inference execution simulation to visualize schedule";

    let description = [{
        Simulate the schedule generated by the compiler and using the cost model visualize inference execution.
    }];

    let constructor = "vpux::VPURT::createInferenceExecutionAnalysisPass()";
}

#endif
