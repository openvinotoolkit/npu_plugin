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

    let description = [{
        The pass adjusts workload size for NCEDepthConvolution, NCEMaxPool, NCEAveragePool and NCEPermuteQuantize,
        as well as for NCE operations that produce sparse activations.

        NCEDepthConvolutionOp, NCEMaxPoolOp and NCEAveragePoolOp require the number of channels to be 16, 32 or 64.
        If the number of channels does not match, workload is split.

        NCEPermuteQuantizeOp rotates output dimensions and padding may be used to indicate the expansion over height.
        It is necessary to subtract pads from respective workload dimensions and then set zero padding.

        NCE operations with sparse outputs must have all variants with the same number of channels and the number
        of channels has to be a power of two. Additionally, if the NCE op shares a consumer with another NCE op
        (directly or indirectly), the number of channels of their variants must be aligned.
    }];

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


//
// DetectInPlaceEltwise
//

def DetectInPlaceEltwise : PassBase<"detect-in-place-eltwise", "vpux::FunctionPass"> {
    let summary = "Convert Eltwise operation to read and write to the same buffer in memory";

    let description = [{
        This pass will check if Eltwise operation has input and output buffers of the same size
        in memory and mark such Eltwise eligible for inplace execution.
        It will write the result into one of the inputs in memory.
    }];

    let constructor = "vpux::VPU::createDetectInPlaceEltwisePass()";
}


//=================================================================================
// Sparsity
//=================================================================================

//
// WrapOpsInSparsifyDesparsifyPairs
//

def WrapOpsInSparsifyDesparsifyPairs : PassBase<"wrap-ops-in-sparsify-pairs", "vpux::FunctionPass"> {
    let summary = "Wrap operations in pairs of Sparsify-Desparsify";

    let description = [{
        Wraps operations in pairs of Sparsify-Desparify ops. The sparsity profile
        will determine which operations will be wrapped:
        - profile S0: add SparsifyOp for each input and Sparsify-Desparsify chain for output
        - profile S1: add Sparsify-Desparsify chain both for inputs and output
    }];

    let constructor = "vpux::VPU::createWrapOpsInSparsifyDesparsifyPairsPass(vpux::VPU::symbolizeEnum<VPU::ActivationSparsityProfile>)";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "sparsityProfile", "sparsity-profile",
            "std::string", [{""}],
            "Flag to choose sparsity profile"
        >
    ];
}

//
// FuseSparsityOps
//

def FuseSparsityOps : PassBase<"fuse-sparsity-ops", "vpux::FunctionPass"> {
    let summary = "Fuse subsequent [De]SparsifyOps with SparseOpInterface ops";

    let constructor = "vpux::VPU::createFuseSparsityOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "fuseSparsify", "fuse-sparsify",
            "bool", "false",
            "Flag to choose inputs or output will be handled"
        >
    ];
}

//
// OptimizeSparsifyDesparsifyPairs
//

def OptimizeSparsifyDesparsifyPairs : PassBase<"optimize-sparsify-desparsify-pairs", "vpux::FunctionPass"> {
    let summary = "Optimize common patterns of subsequent sparsify-desparsify ops to remove redundant conversions";

    let constructor = "vpux::VPU::createOptimizeSparsifyDesparsifyPairsPass(vpux::VPU::symbolizeEnum<VPU::ActivationSparsityProfile>)";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "sparsityProfile", "sparsity-profile",
            "std::string", [{""}],
            "Flag to choose sparsity profile"
        >
    ];
}

//
// OptimizeSparsityOps
//

def OptimizeSparsityOps : PassBase<"optimize-sparsity-ops", "vpux::FunctionPass"> {
    let summary = "Optimize additional sparsity patterns";

    let description = [{
        Some optimizations such duplicated Sparsify ops for Eltwise, first Sparsify
        or last Desparsify cant be done during WrapOpsInSparsifyDesparsifyPairs pass
        until output sparsity wouldnt be fused
    }];

    let constructor = "vpux::VPU::createOptimizeSparsityOpsPass(vpux::VPU::symbolizeEnum<VPU::ActivationSparsityProfile>)";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "sparsityProfile", "sparsity-profile",
            "std::string", [{""}],
            "Flag to choose sparsity profile"
        >
    ];
}

//
// LowerSparsityOps
//

def LowerSparsityOps : PassBase<"lower-sparsity-ops", "vpux::FunctionPass"> {
    let summary = "Convert Sparsify/Desparsify ops to Eltwise or GroupSparseBufferOp";

    let constructor = "vpux::VPU::createLowerSparsityOpsPass()";

    let description = [{
        Convert left Sparsify/Desparsify operations to actual HW ops. Desparsify converts
        to Eltwise with aliased inputs, while Sparsify lowering controled by fake-sparsify option.
        If fake-sparsify enabled lowering, then fake sparsity map(all values are 1's) will be generated.
        Otherwise lowering in same way as Desparsify.
    }];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "fakeSparsify", "fake-sparsify",
            "bool", "false",
            "Flag to choose method of VPU.Sparsify lowering"
        >
    ];
}

//
// SparsifyWeights
//

def SparsifyWeights : PassBase<"sparsify-weights", "vpux::FunctionPass"> {
    let summary = "Sparsify weights for NCE ops";

    let description = [{
        Convert const parameters for NCE ops to sparse types depending on sparsify strategy.
    }];

    let constructor = "vpux::VPU::createSparsifyWeightsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

}

//
// RecomputeSparsityPtrs
//

def RecomputeSparsityPtrs : PassBase<"recompute-sparsity-ptrs", "vpux::FunctionPass"> {
    let summary = "Recomputes sparsity pointers";

    let description = [{
        Recomputes the sparsity pointers inside the weights table for sparse weights.
    }];

    let constructor = "vpux::VPU::createRecomputeSparsityPtrsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

}

//
// AddSparsityMapToSparseActivations
//

def AddSparsityMapToSparseActivations : PassBase<"add-sparsity-map-to-sparse-activations", "vpux::FunctionPass"> {
    let summary = "Update type of result for operations which produce SparseTensor type.";

    let description = [{
        Pass updates output type of operations which produce sparsified output. It adds sparsity_map to output tensor type.
        Then it propagates type to all users until sparse data consumer is reached.
    }];

    let constructor = "vpux::VPU::createAddSparsityMapToSparseActivationsPass()";

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
    let summary = "Sets activation function for VPU37XX PPE based on clamp range";

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

//
// EnsureNCEOpsSizeRequirements
//

def EnsureNCEOpsSizeRequirements : PassBase<"ensure-nce-ops-size-requirements", "vpux::FunctionPass"> {
    let summary = "Ensure hw operations meet size requirements";

    let description = [{
        This pass ensures that hardware operations meet hardware size requirements:
        each operation need to have less than 8192 values per dimension. This is done
        by tiling such operations into smaller ones.
        
        Note: currently, operations with input channels greater than 8192 will cause this
        pass to fail. I introduced allowLargeInputChannels config, in order to bypass this failure for Model_G,
        this will be removed when input channel implementation done.
    }];

    let constructor = "vpux::VPU::createEnsureNCEOpsSizeRequirementsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// FuseClampPass
//

def FuseClampPass : PassBase<"fuse-clamp", "vpux::FunctionPass"> {
    let summary = "Fuses VPU.Clamp parameters into previous NCE operation";

    let description = [{
        This pass follows `SetupPPEPass` and fuses VPU.Clamp with already existing PPE task.
        1. Search for VPU.NCE -> VPU.Clamp pattern
        2. Fetch min and max parameters from VPU.Clamp
        3. Set clamp_low and clamp_high according to min, max and existing activation
        4. Remove VPU.Clamp from the graph
    }];

    let constructor = "vpux::VPU::createFuseClampPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
