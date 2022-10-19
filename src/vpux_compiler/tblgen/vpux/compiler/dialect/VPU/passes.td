//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPU_PASSES
#define VPUX_COMPILER_DIALECT_VPU_PASSES

include "mlir/Pass/PassBase.td"

//
// InitCompiler
//

def InitCompiler : PassBase<"init-compiler", "vpux::ModulePass"> {
    let summary = "Initializes compiler for VPU platforms";

    let description = [{
        This pass attaches VPU related compilation parameters to Module attributes and
        initializes **IERT Dialect** run-time resources information.
    }];

    let constructor = "vpux::VPU::createInitCompilerPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", [{"VPUX30XX"}],
            "VPU architecture to compile for"
        >,
        Option<
            "compilationModeOpt", "compilation-mode",
            "std::string", [{"DefaultHW"}],
            "[Optional] Set compilation mode as `ReferenceSW`, `ReferenceHW` or `DefaultHW`"
        >,
        Option<
            "numberOfDPUGroupsOpt", "num-of-dpu-groups",
            "int", "",
            "[Optional] Number of available DPU groups"
        >,
        Option<
            "numberOfDMAPortsOpt", "num-of-dma-ports",
            "int", "",
            "[Optional] Number of available DMA ports"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "vpux::VPU::VPUDialect"
    ];
}

//
// Multi-cluster strategy assignment
//

def MultiClusterStrategyAssignment : PassBase<"multi-cluster-strategy-assignment", "vpux::FunctionPass"> {
    let summary = "This pass compute the hardware efficiency of layer that is executed as SOH or SOK and assigns the most optimal strategy";

    let constructor = "vpux::VPU::createMultiClusterStrategyAssignmentPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Wrap multi-cluster layers in NCEClusterTiling
//

def WrapVPUOpsInNCEClusterTiling : PassBase<"wrap-vpu-ops-in-ncecluster-tiling", "vpux::FunctionPass"> {
    let summary = "This pass wraps vpu operations that should be executed across multiple clusters in NCEClusterTiling operations";

    let description = [{
        This pass builds an IR in order to represent multi-cluster compilation. It performs a number of functions.
        1) It creates variations of distributed tensors depending on the multi-cluster strategy of the layer. 
        2) It creates DMA operations DDR->CMX and wraps the DMAs in NCEClusterTiling.
        3) It wraps hardware executable operations in NCEClusterTiling. 
    }];

    let constructor = "vpux::VPU::createWrapVPUOpsInNCEClusterTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Manual strategy utils
//

def ManualStrategyUtils : PassBase<"manual-strategy-utils", "vpux::FunctionPass"> {
    let summary = "Utils for reading or writing a json strategy";

    let description = [{
        Utility allowing to store and write as JSON the current selected strategy from the two strategy passes
        createMultiClusterStrategyAssignmentPass() and createPrefetchTilingPass(). And also to manually 
        overwrite the strategy.
    }];

    let constructor = "vpux::VPU::createManualStrategyUtilsPass()";

    let options = [
        Option<
            "writeStrategyToJSON", "write-strategy-to-json",
            "bool", "false",
            "Flag to enable writing strategy to file"
        >,
        Option<
            "writeStrategyFileLocation", "write-strategy-file-location",
            "std::string", [{"strategy.json"}],
            "Location/path to write strategy file"
        >,
        Option<
            "readStrategyFromJSON", "read-strategy-from-json",
            "bool", "false",
            "Flag to enable reading strategy from file"
        >,
        Option<
            "readStrategyFileLocation", "read-strategy-file-location",
            "std::string", [{"strategy.json"}],
            "Location/path to read strategy file"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitNCEOpsOntoWorkloads
//

def SplitNCEOpsOntoWorkloads : PassBase<"split-NCE-ops-onto-workloads", "vpux::FunctionPass"> {
    let summary = "Split VPU NCE operation onto workloads";

    let constructor = "vpux::VPU::createSplitNCEOpsOntoWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// CorrectNCEWorkloads
//

def CorrectNCEWorkloads : PassBase<"correct-NCE-workloads", "vpux::FunctionPass"> {
    let summary = "Correct NCE workloads if they do not fit requirements";

    let constructor = "vpux::VPU::createCorrectNCEWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AdjustMemorySpace
//

def AdjustMemorySpace : PassBase<"adjust-memory-space", "vpux::FunctionPass"> {
    let summary = "Adjusts the tensor location for VPU-driven operations";

    let description = [{
        The pass adjusts the location of tensors that are used by hardware-driven operations

        Currently, it surrounds VPU-driven nodes with Copy operations to specify that all the data
        that they consume/produce must reside in CMX
    }];

    let constructor = "vpux::VPU::createAdjustMemorySpacePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// FuseM2IOps
//

def FuseM2IOps : PassBase<"fuse-m2i-ops", "vpux::FunctionPass"> {
    let summary = "Fuse M2I ops to VPU::M2ITask";

    let constructor = "vpux::VPU::createFuseM2IOpsPass()";

    let dependentDialects = [
       "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertM2IOps
//

def ConvertM2IOps : PassBase<"convert-m2i-ops", "vpux::FunctionPass"> {
    let summary = "Convert individual M2I ops to VPU::M2ITask";

    let constructor = "vpux::VPU::createConvertM2IOpsPass()";

    let dependentDialects = [
       "vpux::VPU::VPUDialect"
    ];
}

//
// CMXConcat
//

def CMXConcat : PassBase<"cmx-concat", "vpux::FunctionPass"> {
    let summary = "Move Concat operations from DDR to NNCMX";

    let constructor = "vpux::VPU::createCMXConcatPass()";

    let description = [{
        This pass will try to check if a Concat operation can fit in NNCMX
        with few restrictions and if so move the concat from DDR to NNCMX.
    }];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ResolvePWLPostOps
//

def ResolvePWLPostOps : PassBase<"resolve-pwl-post-ops", "vpux::FunctionPass"> {
    let summary = "Resolve requirements for fused PWL post-ops";

    let description = [{
        Ensures the correct quantization ranges are used for fused PWL activation functions.
    }];

    let constructor = "vpux::VPU::createResolvePWLPostOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertScalarToTensor
//

def ConvertScalarToTensor : PassBase<"convert-scalar-to-tensor", "vpux::FunctionPass"> {
    let summary = "Convert a scalar input to tensor";

    let description = [{
        Some operations (e.g. Gather) do not support scalar data. This pass converts scalar operands to tensors with one element.
    }];

    let constructor = "vpux::VPU::createConvertScalarToTensorPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//=================================================================================
// Tiling
//=================================================================================

//
// Isolated Tiling
//

def IsolatedTiling : PassBase<"isolated-tiling", "vpux::FunctionPass"> {
    let summary = "Tile layers in isolation so that all their I/O meet the memory capacity";

    let description = [{
        The pass applies tiling to the layers whose memory requirements exceed the capacity available.

        The pass tries to split each single layer in isolation, with no smarter heuristics
        such as "allow running in parallel" or "allow continious computation in tiles" or any else.

        The pass does not use any cost model to optimize the entire layer's processing time. It just
        iteratively increases the number of tiles until the the largest tile's memory requirements  meet
        the device capacity, and stops there.
    }];

    let constructor = "vpux::VPU::createIsolatedTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Prefetch Tiling
//

def PrefetchTiling : PassBase<"prefetch-tiling", "vpux::FunctionPass"> {
    let summary = "Tile layers into smaller tiles to enable prefetch pipeline";

    let description = [{
        The pass performs tiling on layers to enable prefetch pipeline.

        The pass tries run tiles in parallel.
        The 'prefetch' means that the next tile could be loaded in advance when the current tile is computing.

        The pass does not consider cost models,
        only tiles layers to make at least two tiles could be loaded in CMX memory at the same time.
    }];

    let constructor = "vpux::VPU::createPrefetchTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// OptimizeConcatSliceToSliceConcat
//

def OptimizeConcatSliceToSliceConcat : PassBase<"optimize-concate-slice-to-slice-concat", "vpux::FunctionPass"> {
    let summary = "Optimize concate-slice to slice-concat";

    let description = [{
        This pass optimize concat-slice to slice-concat to reduce data copy.
    }];

    let constructor = "vpux::VPU::createOptimizeConcatSliceToSliceConcatPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Manual Tiling
//

def ManualTiling : PassBase<"manual-tiling", "vpux::FunctionPass"> {
    let summary = "Tile layers with manual strategy";

    let description = [{
        The pass performs manual tiling on layers specified by the user.
    }];

    let constructor = "vpux::VPU::createManualTilingPass()";
}

//
// SetupPPEPass
//

def SetupPPEPass : PassBase<"setup-ppe", "vpux::FunctionPass"> {
    let summary = "Sets activation function for VPU37XX and VPUX40XX PPE based on clamp range";

    let description = [{
        Ensures the correct activation function and clamping is used for PPE.
        Namely:
        * When ReLU shift value is non-zero, set leaky ReLU.
        * Otherwise, set NOOP.
        * Deduce clamping via output element type.
    }];

    let constructor = "vpux::VPU::createSetupPPEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
