//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
            "std::string", "",
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
        >,
        Option<
            "allowCustomValues", "allow-custom-values",
            "bool", "",
            "[Optional] Allows keep predefined values in IR"
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

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`. `PREFETCH` is set by default"
        >
    ];

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

    let options = [
        Option<
            "enableExplicitDistributedTensorAttr", "enable-explicit-distributed-attr",
            "bool", "false",
            "Flag to enable generating explicit DistributedTensorAttr for Distributed data type."
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Tile over H for the following vertical fusion
//

def TileOverHForVF : PassBase<"tile-over-h-for-vf", "vpux::FunctionPass"> {
    let summary = "Assign tiling strategy over H for the following vertical fusion";

    let description = [{
        This pass tiles candidate vertical fusion operations over H for the following vertical fusion passes.
    }];

    let constructor = "vpux::VPU::createTileOverHForVFPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", "",
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >
    ];
}

//
// Wrap operations in VerticalFusionOp
//

def WrapVerticalFusionRegion : PassBase<"wrap-in-vertical-fusion", "vpux::FunctionPass"> {
    let summary = "This pass wraps vpu operations that might be tiled in order to implement VF";

    let description = [{
        Wrap operations to VerticalFusion block which match criterias
        1. Operation has activation tiling (activation doesn't fit in CMX)
        2. NCE operations don't have strides larger than 1
        3. Even if NCE operation doesn't have activation tiling, but its kernel is 1x1,
        it also might be wrapped because there is no additional computation cost of it
    }];

    let constructor = "vpux::VPU::createWrapVerticalFusionRegionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Move view like operations to VF
//

def MoveViewOpsToVF : PassBase<"move-view-ops-to-vf", "vpux::FunctionPass"> {
    let summary = "Move view like operations to VF subgraphs";

    let description = [{
        Move view like operations to nearest VF subgraphs,
        which allows to build larger ones on the next step.
    }];

    let constructor = "vpux::VPU::createMoveViewOpsToVerticalFusionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Build VF Subgraph
//

def MergeVfSubgraphs : PassBase<"merge-vertical-fusion-subgraphs", "vpux::FunctionPass"> {
    let summary = "Build subgraph from VF single regions";

    let description = [{
        Merge VF blocks and add operations to them recalculating tiling information
        in following cases:
        1. Number of operations which might increase computational cost does not exceed limit
        2. All operations have same multicluster strategy or don't have them at all
        3. Region which is supposed to be added doesn't have any other users except current region or
        all its users point to current region too.
        4. All operations in new region after merging fit in CMX when they are tiled for VF. In case they don't, number of tiles
        increases.
        5. Required CMX memory by constant weights shouldn't exceed the threshold to avoid spilling.
    }];

    let constructor = "vpux::VPU::createMergeVfSubgraphsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableVerticalFusionPipelining", "enable-vertical-fusion-pipelining",
            "bool", "false",
            "Flag to enable vertical fusion pipelining"
        >
    ];
}

//
// VF Tiling
//

def VfTiling : PassBase<"vertical-fusion-tiling", "vpux::FunctionPass"> {
    let summary = "Apply tiling on VF subgraph";

    let description = [{
        Apply VF tiling on subgraph wrapped in VF region.
    }];

    let constructor = "vpux::VPU::createVfTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableVerticalFusionPipelining", "enable-vertical-fusion-pipelining",
            "bool", "false",
            "Flag to enable vertical fusion pipelining"
        >
    ];
}

//
// Unroll unused VerticalFusionOp
//

def UnrollUnusedVerticalFusionRegion : PassBase<"unroll-unused-vertical-fusion", "vpux::FunctionPass"> {
    let summary = "Unroll single VF blocks";

    let description = [{
        Unroll VF block in case it hasn't been assembled with other blocks in subgraph
    }];

    let constructor = "vpux::VPU::createUnrollUnusedVerticalFusionRegionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// Roll back the H-prioritized tiling strategy if unused
//

def RollBackTilingStrategy : PassBase<"roll-back-tiling-strategy", "vpux::FunctionPass"> {
    let summary = "Roll back the H-prioritized tiling strategy if unused";

    let description = [{
        If the tiling strategy is changed to H dimension to enable vertical fusion but at last not vertical-fused
        roll back to the original tiling strategy
    }];

    let constructor = "vpux::VPU::createRollBackTilingStrategyPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", "",
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >
    ];
}

//
// Adjust tiling strategy for VF
//

def AdjustVFTilingStrategy : PassBase<"adjust-vf-tiling-strategy", "vpux::FunctionPass"> {
    let summary = "Adjust tiling strategy in order to make sure that all subgraphs fit in CMX";

    let description = [{
        Take into account conditions in order to avoid additional spills
        In case some of them breaks, increase number of tiles
        Following buffers should fit in CMX for each VF tile
        1. Input tensors of VF tile
        2. Output of VF tile
        3. Largest operation in the block
    }];

    let constructor = "vpux::VPU::createAdjustVFTilingStrategyPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableVerticalFusionPipelining", "enable-vertical-fusion-pipelining",
            "bool", "false",
            "Flag to enable vertical fusion pipelining"
        >
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
// Strategy manager pass
//

def StrategyManagerImpl : PassBase<"strategy-manager", "vpux::FunctionPass"> {
    let summary = "Assignment and optimization multi-cluster strategies to operations";

    let description = [{
        Pass consists of two parts:
        1. Assignment of multicluster strategies and tiling strategies to each operation based on vpunn cost of each strategy.
        2. Optimization/adjustment of strategies based on one of common optimization algorithm.
    }];

    let constructor = "vpux::VPU::createStrategyManagerImplPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`. `PREFETCH` is set by default"
        >
    ];
}

//
// ResolveEltwiseWithZTiledWorkloads
//

def ResolveEltwiseWithZTiledWorkloads : PassBase<"resolve-eltwise-with-z-tiled-workloads", "vpux::FunctionPass"> {
    let summary = "Resolves Eltwise operations which have workloads tiled over Z";

    let description = [{
        Hardware Eltwise does not support variants tiled over the Z dimension. If such cases are encountered,
        these operations are split into separate Eltwise operations, each containing the workloads that cover
        a different subset of channels.

        For example, if the original Eltwise contains the following workloads:
            1. offset = [0, 0,  0, 0], sizes = [1, 64, 8, 16], cluster_id = 0
            2. offset = [0, 64, 0, 0], sizes = [1, 64, 8, 16], cluster_id = 0
            3. offset = [0, 0,  8, 0], sizes = [1, 64, 8, 16], cluster_id = 1
            4. offset = [0, 64, 8, 0], sizes = [1, 64, 8, 16], cluster_id = 1
        Two Eltwise operations will be created, the first one containing workloads 1 and 3, the other one
        workloads 2 and 4, with their channel offsets reset to zero. The correct subset of channels is
        sliced individually for each new Eltwise operation.

        In case the inputs are distributed types in CMX, manual copy operations that spill them to DDR are
        introduced, in order to avoid Slice operations that work with these types. These Slice operations
        would get lowered to copies where both the input and output are distributed types; such scenarios
        are not fully supported (E#78676).

        The outputs of the smaller Eltwise operations get copied to DDR in order to avoid accuracy degradation
        that takes place when the outputs are concatenated in CMX.
    }];

    let constructor = "vpux::VPU::createResolveEltwiseWithZTiledWorkloadsPass()";

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
// SplitGRUSequence
//

def SplitGRUSequence : PassBase<"split-gru-sequence", "vpux::FunctionPass"> {
    let summary = "Split GRUSequence into GRUSequenceFirstPart and GRUSequenceLastPart";

    let description = [{
        The pass can split GRUSequence into two parts to fit into CMX when tiling strategy can't be generated.
    }];

    let constructor = "vpux::VPU::createSplitGRUSequencePass()";

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

    let constructor = "vpux::VPU::createWrapOpsInSparsifyDesparsifyPairsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "enableActivationSparsityMode", "enable-activation-sparsity-mode",
            "std::string", [{"false"}],
            "Activation sparsity enablement mode (auto, true or false)"
        >,
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
        Converts Sparsify operations to Convolutions and Desparsify operations to Eltwise ops.

        In case the `fakeSparsity` flag is set to true, Sparsify operations are instead converted to a
        GroupSparseTensor operation whose sparsity map contains only values of 1. This lets the data be
        interpreted as a sparse one without actually removing the sparse values.
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

//
// SplitSEOps
//

def SplitSEOps : PassBase<"split-se-ops", "vpux::FunctionPass"> {
    let summary = "Split compatible SE operations for better performance";

    let description = [{
        Finds operations that can be executed on hardware using the Storage Element pointers
        feature and splits them if they have benefits for performance.

        The list of supported operations:
        Interpolate Op that satisfied the following limitations:
        - The NCEInterpolate size is larger than the CMX size;
        - Both factors on H and W are larger than 4
    }];

    let constructor = "vpux::VPU::createSplitSEOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// LowerOpsToSENCE
//

def LowerOpsToSENCE : PassBase<"lower-ops-to-se-nce", "vpux::FunctionPass"> {
    let summary = "Converts compatible operations to SE NCE operations";

    let description = [{
        Finds operations that can be executed on hardware using the Storage Element pointers
        feature and lowers them to VPU.NCE.

        The list of supported operations:
        - Interpolate - mode: NEAREST
                        axes: H, W
                        coordinate_transformation_mode: all (except ALIGN_CORNERS)
                        nearest_mode: all
                        scale: integer only
                        padding: none

                      - mode: LINEAR, LINEAR_ONNX
                        axes: H, W
                        coordinate_transformation_mode: ASYMMETRIC, PYTORCH_HALF_PIXEL, HALF_PIXEL
                        scale: integer only
                          - values between [1-5] for PYTORCH_HALF_PIXEL and HALF_PIXEL
                            when the scale is an even number
                          - values between [1-11] for all other cases
                        padding: none
    }];

    let constructor = "vpux::VPU::createLowerOpsToSENCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// FuseNCEInterpolateConsumers
//

def FuseNCEInterpolateConsumers : PassBase<"fuse-nce-interpolate-consumers", "vpux::FunctionPass"> {
    let summary = "Fuses NCE.Interpolate into consumer NCE.Convolution";

    let description = [{
        Fuses NCE.Interpolate with consumer NCE.Convolution.

        NCE.Interpolate with mode "nearest" is lowered to a dummy NCE.Convolution with
        SE table that upsamples tensor.

        We can simply pass the SE table to this consumer and avoid the dummy convolution.
    }];

    let constructor = "vpux::VPU::createFuseNCEInterpolateConsumersPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//=================================================================================
// Tiling
//=================================================================================

//
// Tiling Strategy Assignment
//

def TilingStrategyAssignment : PassBase<"tiling-strategy-assignment", "vpux::FunctionPass"> {
    let summary = "Assign tiling strategy for layers applicable";

    let description = [{
        The pass assigns tiling strategy for layers whose memory requirements exceed the capacity available.
        The pass only assigns strategy and do not perform any tiling actions, and if tiling strategy is set by
        ManualStrategyUtilsPass, it will not be processed by this pass.

        Isolated tiling: split each single layer in isolation, with no smarter heuristics such as
                         "allow running in parallel" or "allow continious computation in tiles" or any else.
        Prefetch tiling: tries to run tiles in parallel, and 'prefetch' means that the next tile could be loaded
                         in advance when the current tile is computing.

        The pass does not use any cost model to optimize the entire layer's processing time.
    }];

    let constructor = "vpux::VPU::createTilingStrategyAssignmentPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >,
        Option<
            "vpunnCost", "vpunn-cost",
            "bool", "false",
            "Use VPUNN cost model to get the best tiling strategy"
        >
    ];
}

//
// Apply Tiling
//

def ApplyTiling : PassBase<"apply-tiling", "vpux::FunctionPass"> {
    let summary = "Apply tiling on layers with assigned tiling strategy";

    let description = [{
        The pass applies tiling strategy on layers with previously assigned strategy attribute.
    }];

    let constructor = "vpux::VPU::createApplyTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DetectionOutput decomposition
//

def DetectionOutputDecomposition : PassBase<"detection-output-decomposition", "vpux::FunctionPass"> {
    let summary = "Replace DetectionOutput operation with a subgraph of smaller operations";

    let description = [{
        Replace DetectionOutput operation
        ┌─────────┐   ┌────────────────┐  ┌──────────┐
        │BoxLogits│   │ClassPredictions│  │PriorBoxes│
        └────┬────┘   └───────┬────────┘  └─────┬────┘
             │                │                 │
             │         ┌──────┴────────┐        │
             └─────────┤DetectionOutput├────────┘
                       └───────────────┘

        with a subgraph (Reshapes and MemPermutes are ommited)
        ┌─────────┐  ┌──────────┐        ┌────────────────┐
        │BoxLogits│  │PriorBoxes│        │ClassPredictions│
        └───────┬─┘  └─┬────────┘        └───────┬────────┘
                │      │                         │
        ┌───────┴──────┴───────────┐  ┌──────────┴────────────┐
        │DetectionOutputDecodeBoxes│  │DetectionOutputSortTopK│
        └───────────────────┬──────┘  └───┬──┬─────┬─┬────────┘
                            │             │  │     │ │
                      ┌─────┴─────────────┴──┴───┐ │ │
                      │DetectionOutputSelectBoxes│ │ │
                      └─────────────┬────────────┘ │ │
                                    │              │ │
                                  ┌─┴──────────────┴─┴────┐
                                  │DetectionOutputNmsCaffe│
                                  └────┬─┬─┬──────────────┘
                                       │ │ │
                          ┌────────────┴─┴─┴────────────┐
                          │DetectionOutputCollectResults│
                          └─────────────────────────────┘
    }];

    let constructor = "vpux::VPU::createDetectionOutputDecompositionPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
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

//
// AdjustTilingForPermuteQuantizePass
//

def AdjustTilingForPermuteQuantizePass : PassBase<"adjust-tiling-for-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Adjust Slice and Concat position after tiling permute quantize";

    let description = [{
        This pass rewrites the operation sequence for `VPU.NCE.PermuteQuantize`.
        After `ApplyTilingPass` `VPU.Slice` and `VPU.Concat` operations will be inserted next to `VPU.NCE.PermuteQuantize`,
        but we want to slice and concatenate all the sequence (Reshape -> LayoutCast -> PermuteQuantize -> LayoutCast -> [optional Reshape]).
    }];

    let constructor = "vpux::VPU::createAdjustTilingForPermuteQuantizePass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// OptimizeConcat
//

def OptimizeConcat : PassBase<"optimize-concat", "vpux::FunctionPass"> {
    let summary = "Try to eliminate Concat for Concat-Slice pattern";

    let description = [{
        After `ApplyTilingPass` lots of `VPU.Concat`-`VPU.Slice` are introduced, `VPU.Concat` can be eliminated only if all its users are `VPU.Slice` and
        input tensor of each `VPU.Slice` is actually sub-tensor of one of input tensors of Concat.
    }];

    let constructor = "vpux::VPU::createOptimizeConcatPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif

//
// RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass
//

def RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass : PassBase<"remove-output-sparse-to-avoid-suboptimal-dpu-workloads", "vpux::FunctionPass"> {
    let summary = "Remove output sparsity for SOK layer to avoid suboptimal dpu workloads";

    let description = [{
        This pass removes SOK layer's output sparsity if
        1. SOK layer has different split sizes on clusters excluding the last one. For example, we need to split OC = 128 on 6 tiles, 
        the tiled size will be {32, 32, 16, 16, 16, 16}. If there's output sparsity, we need to split 32 into two pieces of 16 because 
        we must have the same workload channel excluding the last one. However, two workloads with 16 channels have much worse
        performance than a workload with 32 channels. If there's no sparsity, we can keep the workload with 32 channels. 
        This is only relevant to the senario with more than 2 tiles so it doesn't exist for VPU37XX.
        2. SOK layer's output is used by `VPU.Concat`
        
        Conv1_1 (OC = 256, SOK)  Conv1_2 (OC = 256, SOK)
             \                               /
                         Concat on C
                              |
                            Conv2
      
       Take above graph as an example, we need to split OC = 256 on 6 tiles, the tiled size will be {48, 48, 48, 48, 48, 16}. 
       After concatenation, the combined workloads will be {48, 48, 48, 48, 48, 16, 48, 48, 48, 48, 48, 16}. If there's output sparsity for Conv1_1 and Conv1_2, 
       we need to split 48 into three pieces of 16 because we must have the same workload channel excluding the last one. If there's no sparsity, we can
       keep the workload with 48 channels.
    }];

    let constructor = "vpux::VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}
