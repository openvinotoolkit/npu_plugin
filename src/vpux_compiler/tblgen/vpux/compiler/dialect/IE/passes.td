//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_IE_PASSES
#define VPUX_COMPILER_DIALECT_IE_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// Precisions and Layouts
//=================================================================================

//
// UseUserPrecision
//

def UseUserPrecision : PassBase<"use-user-precision", "vpux::ModulePass"> {
    let summary = "Use user precisions for entry point function prototype";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass updates the CNNNetwork entry point function prototype and use user-provided precisions for its operands and results.
        The pass inserts Convert operations from/to topology precisions.
    }];

    let constructor = "vpux::IE::createUseUserPrecisionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UseUserLayout
//

def UseUserLayout : PassBase<"use-user-layout", "vpux::ModulePass"> {
    let summary = "Use user layouts for entry point function prototype";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass updates the CNNNetwork entry point function prototype
        and uses user-provided layouts for its operands and results.
        The pass inserts Reorder operations from/to topology layout.
    }];

    let constructor = "vpux::IE::createUseUserLayout()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustLayouts
//

def AdjustLayouts : PassBase<"adjust-layouts", "vpux::FunctionPass"> {
    let summary = "Adjust required layouts for all layers";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass adds the required layouts instead of the default one
        depending on the layer specification from underlying Dialect.
    }];

    let constructor = "vpux::IE::createAdjustLayoutsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ForceHostQuantization
//

def ForceHostQuantization : PassBase<"force-host-quantization", "vpux::ModulePass"> {
    let summary = "Support input quantization on host side for VPUX compiler";

    let description = [{
        Support input quantization on host side for VPUX compiler.

        Note: current implementation will not allow us to use this feature in CiD case.
    }];

    let constructor = "vpux::IE::createForceHostQuantizationPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeReorders
//

def OptimizeReorders : PassBase<"optimize-reorders", "vpux::FunctionPass"> {
    let summary = "Optimize extra Reorder operations";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass tries to optimize out Reorder operations for common cases
        by propagating them from inputs to outputs and merging into layers.
    }];

    let constructor = "vpux::IE::createOptimizeReordersPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ForceHostPrecisionLayoutConversion
//

def ForceHostPrecisionLayoutConversion : PassBase<"force-host-precision-layout-conversion", "vpux::ModulePass"> {
    let summary = "Move pre-/post- processing operations to host side";

    let description = [{
        This pass detects the following pre-/post- processing operations and excludes them from device IR:

        * Convert
        * Reorder

        The pre-/post- processing will be performed on host by the plugin.
    }];

    let constructor = "vpux::IE::createForceHostPrecisionLayoutConversionPass()";
}

//=================================================================================
// AdjustForVPU
//=================================================================================

//
// ConvertAssignReadValueToReturnsAndInputs
//

def ConvertAssignReadValueToReturnsAndInputs : PassBase<"convert-assign-read-value", "vpux::FunctionPass"> {
    let summary = "Convert assign to returns and read value to inputs";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Assign` operations with main function returns and
        `ReadValue` operations with main function inputs.
    }];

    let constructor = "vpux::IE::createConvertAssignReadValueToReturnsAndInputs()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertNearestToStridedConcat
//

def ConvertNearestToStridedConcat : PassBase<"convert-nearest-to-strided-concat", "vpux::FunctionPass"> {
    let summary = "Convert nearest interpolate op to strided concat ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Nearest Interpolate` operations with `Concat` operations with strides.
    }];

    let constructor = "vpux::IE::createConvertNearestToStridedConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertBilinearToStridedConcatAndConv
//

def ConvertBilinearToStridedConcatAndConv : PassBase<"convert-bilinear-to-strided-concat-and-conv", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate op to strided concat, MaxPool and some depthwise convolution Ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Bilinear Interpolate` operations with `Concat` operations with strides, MaxPool and some `depthwise` convolutions.
    }];

    let constructor = "vpux::IE::createConvertBilinearToStridedConcatAndConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPrecisionToFP16
//

def ConvertPrecisionToFP16 : PassBase<"convert-precision-to-fp16", "vpux::ModulePass"> {
    let summary = "Convert tensors precision from FP32 to FP16";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.

        This pass replaces all FP32 tensors with FP16.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToFP16Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPrecisionToI32
//

def ConvertPrecisionToI32 : PassBase<"convert-precision-to-i32", "vpux::ModulePass"> {
    let summary = "Convert tensors precision from I64 to I32";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.
        This pass replaces all I64 tensors with I32.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToI32Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustSoftwareOpsPrecision
//

def AdjustSoftwareOpsPrecision : PassBase<"adjust-software-ops-precision", "vpux::ModulePass"> {
    let summary = "Adjust precision of software ops to satisfy kernel implementation";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.

        Some kernel implementations only support specific precisions. To satisfy this requirement,
        such ops are surrounded by conversion layers.
    }];

    let constructor = "vpux::IE::createAdjustSoftwareOpsPrecisionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDepth2SpaceLayer
//

def ConvertDepth2SpaceLayer : PassBase<"convert-depthToSpace", "vpux::FunctionPass"> {
    let summary = "Convert DepthToSpace layer to {reshape -> transpose -> reshape} subgraph";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `DepthToSpace` operations with {reshape -> transpose -> reshape} subgraph.
    }];

    let constructor = "vpux::IE::createConvertDepth2SpaceLayerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSpace2DepthLayer
//

def ConvertSpace2DepthLayer : PassBase<"convert-spaceToDepth", "vpux::FunctionPass"> {
    let summary = "Convert SpaceToDepth layer to {reshape -> transpose -> reshape} pattern";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `SpaceToDepth` operations with {reshape -> transpose -> reshape} pattern.
    }];

    let constructor = "vpux::IE::createConvertSpace2DepthLayerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGatherToSlice
//

def ConvertGatherToSlice : PassBase<"convert-gather-to-slice", "vpux::FunctionPass"> {
    let summary = "Convert Gather operation to Slice operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces legal `Gather` operations with `Slice` operations.
    }];

    let constructor = "vpux::IE::createConvertGatherToSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertShapeTo4D
//

def ConvertShapeTo4D : PassBase<"convert-shape-to-4d", "vpux::FunctionPass"> {
    let summary = "Convert tensors shapes to 4D";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces ND tensor with 4D analogues for layers, which has such limitations on VPUIP level.
        Also this pass replaces ND network inputs and outputs with 4D analogues to overcome runtime limitations.
    }];

    let constructor = "vpux::IE::createConvertShapeTo4DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPaddingsToFloorMode
//

def ConvertPaddingsToFloorMode : PassBase<"convert-paddings-to-floor-mode", "vpux::FunctionPass"> {
    let summary = "Convert Convolution and Pooling layers paddings to FLOOR rouding mode";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass updates padding attributes for Convolution and Pooling layers.
        It switches layer rounding mode to FLOOR and updates paddings to satisfy output shape.
    }];

    let constructor = "vpux::IE::createConvertPaddingsToFloorModePass()";
}

//
// ConvertTile2PerAxisTile
//

def ConvertTile2PerAxisTile : PassBase<"convert-tile-to-per-axis-tiles", "vpux::FunctionPass"> {
    let summary = "Convert tile op by multiple axes to multiple PerAxisTile operations";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `Tile` op with a set of `PerAxisTile` operations.
    }];

    let constructor = "vpux::IE::createConvertTile2PerAxisTilePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertFCToConv
//

def ConvertFCToConv : PassBase<"convert-fc-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert FullyConnected op to Convolution operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `FullyConnected` operations with `Convolution` operation.
        It inserts extra `Reshape` operations to satisfy `Convolution` specification.
    }];

    let constructor = "vpux::IE::createConvertFCToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertShuffleChannels
//

def ConvertShuffleChannels : PassBase<"convert-shuffle-channels", "vpux::FunctionPass"> {
    let summary = "Convert ShuffleChannels to Reshape->Transpose->Reshape";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        Converts ShuffleChannels to Reshape->Transpose->Reshape.
    }];

    let constructor = "vpux::IE::createConvertShuffleChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollBatch
//

def UnrollBatch : PassBase<"unroll-batch", "vpux::FunctionPass"> {
    let summary = "Split FullyConnected inputs with multiple rows";

    let description = [{
        This pass splits `FullyConnected` inputs by rows.

        For example, `FullyConnected` input with 2x64 geometry will be split by two inputs with 1x64 dimensions.
        Resulting vector rows go to corresponding `FullyConnected` operations and the outputs are concatenated.
    }];

    let constructor = "vpux::IE::createUnrollBatchPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MatMulInputsTo2d
//

def MatMulInputsTo2d : PassBase<"matmul-inputs-to-2d", "vpux::FunctionPass"> {
    let summary = "Convert MatMul inputs to 2d";

    let description = [{
        This pass converts `MatMul` inputs to 2d.

        For example, `MatMul` input with 4x1x64 geometry will be split to four inputs with 1x64 dimensions.
        Resulting inputs with filters go to `MatMul` operations and the outputs are concatenated.
    }];

    let constructor = "vpux::IE::createMatMulInputsTo2dPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertBatchedConvTo1N
//

def ConvertBatchedConvTo1N : PassBase<"convert-batched-conv-to-1n", "vpux::FunctionPass"> {
    let summary = "Convert Convolution with batched input to new one with batch equal to 1";

    let description = [{
        This pass inserts Transpose and Reshape to convert batched input to new one with batch equal to 1

        Original operation:
            Activation: 4x16x1x1 ->
                                    Conv -> 4x5x1x1
            Weights:    5x16x1x1 ->

        New subgraph:
            Activation:4x16x1x1     Weights:5x16x1x1
                |                       |
            Transpose:16x4x1x1          |
                |                       |
            Reshape:1x16x4x1            |
                |                       |
                        Conv:1x5x4x1
                            |
                      Reshape:5x4x1x1
                            |
                     Transpose:4x5x1x1
    }];

    let constructor = "vpux::IE::createConvertBatchedConvTo1NPass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertMatMulToConv
//

def ConvertMatMulToConv : PassBase<"convert-matmul-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert MatMul with 2d 'weights' to convolution";

    let description = [{
        This pass replaces 2d `Matmul` operations with `Convolution` .
    }];

    let constructor = "vpux::IE::createConvertMatMulToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToScaleShift
//

def ConvertToScaleShift : PassBase<"convert-to-scale-shift", "vpux::FunctionPass"> {
    let summary = "Convert Add and Multiply operations to ScaleShift operations";

    let description = [{
        This pass replaces suitable `Add` and `Multiply` operations with `ScaleShift` operations.
    }];

    let constructor = "vpux::IE::createConvertToScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertBroadcastToTile
//

def ConvertBroadcastToTile : PassBase<"convert-broadcast-to-tile", "vpux::FunctionPass"> {
    let summary = "Convert Broadcast operation to Tile operation";

    let description = [{
        This pass replaces `Broadcast` operation with `Tile` operation.
    }];

    let constructor = "vpux::IE::createConvertBroadcastToTilePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSubtractToAdd
//

def ConvertSubtractToAdd : PassBase<"convert-subtract-to-add", "vpux::FunctionPass"> {
    let summary = "Convert Subtract operation to Add with either Negative or DW Conv operations";

    let description = [{
        This pass replaces `Subtract` operation with `Add` with `Negative` operations on VPUX30XX or `Add` with `DW Conv` operations on VPUX37XX.
    }];

    let constructor = "vpux::IE::createConvertSubtractToAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertScaleShiftToDW
//

def ConvertScaleShiftToDW : PassBase<"convert-scale-shift-depthwise", "vpux::FunctionPass"> {
    let summary = "Convert Scale-Shift operation to Depthwise Convolution";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        Convert Scale-Shift operation to Depthwise convolution.
    }];

    let constructor = "vpux::IE::createConvertScaleShiftToDWPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePostOps
//

def FusePostOps : PassBase<"fuse-post-ops", "vpux::FunctionPass"> {
    let summary = "Fuse activation functions with tasks that support post-processing";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Fuse activation functions (e.g. ReLU, leaky ReLU) with tasks that support post-processing
        depending on the compilation mode
    }];

    let constructor = "vpux::IE::createFusePostOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePermuteQuantize
//

def FusePermuteQuantize : PassBase<"fuse-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Converts Quantize-MemPermute combination in 1 common operation";

    let description = [{
        Converts Quantize-MemPermute combination in 1 common operation.
    }];

    let constructor = "vpux::IE::createFusePermuteQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateExpand
//

def PropagateExpand : PassBase<"propagate-expand", "vpux::FunctionPass"> {
    let summary = "Propagate Expand operation in order to fuse it with other layers";

    let description = [{
        Propagate Expand through Eltwise Add in case layers before might be fused with Expand
        in following cases:
        1. PermuteQuntize might be fused with Expand in FusePermuteQuantizeExpand pass
        2. DepthToSpace is used with padded channels' descriptor

    }];

    let constructor = "vpux::IE::createPropagateExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePermuteQuantizeExpand
//

def FusePermuteQuantizeExpand : PassBase<"fuse-permute-quantize-expand", "vpux::FunctionPass"> {
    let summary = "Converts Quantize-MemPermute-Expand combination in 1 common operation";

    let description = [{
        Converts Quantize-MemPermute-Expand combination in 1 common operation.
    }];

    let constructor = "vpux::IE::createFusePermuteQuantizeExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ExpandActivationChannels
//

def ExpandActivationChannels : PassBase<"expand-activation-channels", "vpux::FunctionPass"> {
    let summary = "Allign input tensors shape of DPU operation with hardware requirements";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        This pass processes operations, which can be compile as a DPU tasks and
            expands channels number to number divisible by 16 in case they doesn't satisfy hardware requirements
    }];

    let constructor = "vpux::IE::createExpandActivationChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::StandardOpsDialect"
    ];
}

//
// OptimizeSliceExpand
//

def OptimizeSliceExpand : PassBase<"optimize-slice-expand", "vpux::FunctionPass"> {
    let summary = "Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand in order to avoid extra DMAs
    }];

    let constructor = "vpux::IE::createOptimizeSliceExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LegalizeDilatedConvolution
//

def LegalizeDilatedConvolution : PassBase<"legalize-dilated-conv", "vpux::FunctionPass"> {
    let summary = "Handle dilated convolutions";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        This pass expands filter of dilated convolution so that they are able to be infered
            on dpu because of hardware limitation.
    }];

    let constructor = "vpux::IE::createLegalizeDilatedConvolutionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ResolveStridedSlice
//

def ResolveStridedSlice : PassBase<"resolve-strided-slice", "vpux::FunctionPass"> {
    let summary = "Decouple strided slice to slice + reshape";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        It replaces IE::StridedSlice with non zero masks to a simpler IE::StridedSlice with zero masks + IE::Reshape
        It replaces IE::StridedSlice with dense<1> strides strides with a simple IE::Slice operation
    }];

    let constructor = "vpux::IE::createResolveStridedSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ResolveScatterUpdateByTranspose
//

def ResolveScatterUpdateByTranspose : PassBase<"resolve-scatter-update-by-transpose", "vpux::FunctionPass"> {
    let summary = "Resovle ScatterUpdate operation by Transpose Operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        Only axis == 0 is supported in SWkernel.
        The pass replaces ScatterUpdate(axis!=0) with `IE::Transpose-IE::ScatterUpdate(axis=0)-IE::Transpose` pipeline.

    }];

    let constructor = "vpux::IE::createResolveScatterUpdateByTransposePass()";

    let dependentDialects = [
    "vpux::IE::IEDialect"
    ];
}


//
// ConvertConv1DToConv2D
//

def ConvertConv1DToConv2D : PassBase<"convert-conv1d-to-conv2d", "vpux::FunctionPass"> {
    let summary = "Convert Convolution1D and GroupConvolution1D to its 2D variance";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Extends input, filter and output tensors with height = 1.
        [N, C, W] -> [N, C, 1, W]
        strides:    {2} -> strides:    {1, 2}
        pads_begin: {2} -> pads_begin: {0, 2}
        pads_end:   {2} -> pads_end:   {0, 2}
        dilations:  {2} -> dilations:  {1, 2}
    }];

    let constructor = "vpux::IE::createConvertConv1DToConv2DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UniquifyOps
//

def UniquifyOps : PassBase<"uniquify-ops", "vpux::FunctionPass"> {
    let summary = "Remove duplicating operations with a common producer Value";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass merges operations that are identical to each other, combining consumers.
    }];

    let constructor = "vpux::IE::createUniquifyOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToMemPermute
//

def ConvertToMemPermute : PassBase<"convert-to-mem-permute", "vpux::FunctionPass"> {
    let summary = "Convert Reorder and Transpose ops to MemPermute operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `Reorder` and `Transpose` operations with `MemPermute` operation.
    }];

    let constructor = "vpux::IE::createConvertToMemPermutePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LegalizeNDMemPermute
//

def LegalizeNDMemPermute : PassBase<"legalize-nd-mem-permute", "vpux::FunctionPass"> {
    let summary = "Legalize MemPermute operation with input rank > 4";

    let description = [{
        This pass tries to legalize MemPermute operations by merging dims that are adjacent before and after the permutation.
        Applied only for VPUX.37XX because SW Kernel Tiling is limited to 4D.
    }];

    let constructor = "vpux::IE::createLegalizeNDMemPermutePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SplitConvWithMultipleFQ
//

def SplitConvWithMultipleFQ : PassBase<"split-conv-with-multiple-fq", "vpux::FunctionPass"> {
    let summary = "Splits Convolution for multiple FakeQuantize";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It splits `Convolution` operation with multiple consumers with `FakeQuantize` operations,
        into multiple `Convolution` operations, one for each consumer. This transformation is needed to be
        able to quantize convolution and fuse bias and post-processing operations.
    }];

    let constructor = "vpux::IE::createSplitConvWithMultipleFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDeconvToConv
//

def ConvertDeconv2DToConv2D : PassBase<"convert-deconv-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert Deconvolution 2D to Convolution 2D";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Replaces deconvolution by upsampling and convolution
    }];

    let constructor = "vpux::IE::createConvertDeconv2DToConv2DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleLargeStrides
//

def HandleLargeStrides : PassBase<"handle-large-strides", "vpux::FunctionPass"> {
    let summary = "Handle operations with large strides";

    let description = [{
        This pass splits operations with strides larger than supported on hardware.
    }];

    let constructor = "vpux::IE::createHandleLargeStridesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapMaxPoolWithActivation
//

def SwapMaxPoolWithActivation : PassBase<"swap-maxpool-with-act", "vpux::FunctionPass"> {
    let summary = "Swaps the MaxPool and activation";

    let description = [{
        This pass is needed for VPUX37XX only since HW MaxPool does not support post-op operations.
        Operations are swapped only if there is an operation before MaxPool that supports post-ops.
    }];

    let constructor = "vpux::IE::createSwapMaxPoolWithActivation()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//=================================================================================
// LowPrecision
//=================================================================================

//
// SwapFakeQuantReshape
//

def SwapFakeQuantReshape : PassBase<"swap-fake-quant-reshape", "vpux::FunctionPass"> {
    let summary = "Swap FakeQuantize with Reshape when required to void redundant expand and permute ops";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It matches pattern non-channel-aligned op -> optional Reshapes -> FQ -> Reshapes -> channel-aligned op
        Move the FQ right before the channel-aligned op to avoid redundant expand and permute ops.
    }];

    let constructor = "vpux::IE::createSwapFakeQuantReshapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SplitFakeQuant
//

def SplitFakeQuant : PassBase<"split-fake-quant", "vpux::FunctionPass"> {
    let summary = "Splits FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It splits `FakeQuantize` operations to `quant.qcast -> quant.dcast` pair.
    }];

    let constructor = "vpux::IE::createSplitFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// DequantizeConst
//

def DequantizeConst : PassBase<"dequantize-const", "vpux::FunctionPass"> {
    let summary = "Dequantize constant tensors";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It performs constant folding for `Constant -> quant.dcast` case.
        The pass is used as a fallback to FP16 computations for the cases, where quantized types where not used by layers.
    }];

    let constructor = "vpux::IE::createDequantizeConstPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MergeFakeQuant
//

def MergeFakeQuant : PassBase<"merge-fake-quant", "vpux::FunctionPass"> {
    let summary = "Merge back to FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It merges pair `quant.qcast -> quant.dcast` into single `IE.FakeQuantize`.
        The pass is used as a fallback to FP16 computations for the cases, where quantized types where not used by layers.
    }];

    let constructor = "vpux::IE::createMergeFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DeletePerAxisQuantize
//

def DeletePerAxisQuantization : PassBase<"delete-peraxis-quantization", "vpux::FunctionPass"> {
    let summary = "Delete PerAxis Quantize Dequantize for VPUX37XX";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It deletes per axis quantization which left after LPT.
        Conversion is not mathimatically equal, but for now it gives small
            accuracy deviation
    }];

    let constructor = "vpux::IE::createDeletePerAxisQuantizationPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// PropagateQuantizeDequantize
//

def PropagateQuantizeDequantize : PassBase<"propagate-quantize-dequantize", "vpux::FunctionPass"> {
    let summary = "Propagate Quantize/Dequantize through agnostic operations";

        let description = [{
        The pass is a part of LowPrecision pipeline.

        Quantize/Dequantize are propagated through operations
        }];

let constructor = "vpux::IE::createPropagateQuantizeDequantizePass()";

let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
        ];
}

//
// SwapOperations
//

def SwapOperations : PassBase<"swap-operations", "vpux::FunctionPass"> {
    let summary = "Swap operations implemented ElemTypeInfoOpInterface interface with bias and activation";

        let description = [{

        In order to fuse the bias and activation functions into a main operation,
        intermediate operations will be moved after the fusable operation. For example:
          `Conv -> Reshape -> bias` will be converted to `Conv -> bias -> Reshape`
          `Conv -> Transpose -> ReLU` will be converted to `Conv -> ReLU -> Transpose
        Only operations that implement `IE_ElemTypeInfoOpInterface` are moved.
        Currently, operations are moved only through bias (Add), ReLU, Sigmoid, Tanh.

        }];

let constructor = "vpux::IE::createSwapOperationsPass()";

let dependentDialects = [
        "vpux::IE::IEDialect"
        ];
}

//
// SwapPadLayer
//

def SwapPadLayer : PassBase<"swap-pad-layer", "vpux::FunctionPass"> {
    let summary = "Swap pattern Pad -> Transpose to Transpose -> Pad";

        let description = [{

        In order to fuse Pad layer to Convolution swap Pad with operations between it and Convolution.
        For now only case Pad -> Transpose is supported

        }];

let constructor = "vpux::IE::createSwapPadLayerPass()";

let dependentDialects = [
        "vpux::IE::IEDialect"
        ];
}

//
// FuseQuantizedOps
//

def FuseQuantizedOps : PassBase<"fuse-quantized-ops", "vpux::FunctionPass"> {
    let summary = "Update quantize/dequantize ops";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
    }];

    let constructor = "vpux::IE::createFuseQuantizedOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// RemoveQuantDequantSeq
//

def RemoveQuantDequantSeq : PassBase<"remove-quantdequant-seq", "vpux::FunctionPass"> {
    let summary = "Removes quantize->dequantize ops sequence";

    let description = [{
        The optional pass in the `LowPrecision` pipeline.

        Pass detects pattern quantize -> dequantize and removes it
    }];

    let constructor = "vpux::IE::createRemoveQuantDequantSeqPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// OptimizeUnalignedQDQSeq
//

def OptimizeUnalignedQDQSeq : PassBase<"optimize-unaligned-qdq-seq", "vpux::FunctionPass"> {
    let summary = "Swaps AffineReshape->FakeQuantize sequence if channels become unaligned after AffineReshape";

    let description = [{
        Pass swaps order of AffineReshape->FakeQuantize sequence if channels become unaligned after AffineReshape
        Otherwise additionals ops are introduce in order to align channels which impacts performance.
    }];

    let constructor = "vpux::IE::createOptimizeUnalignedQDQSeqPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// ConvertWeightsToU8
//

def ConvertWeightsToU8 : PassBase<"convert-weights-to-u8", "vpux::FunctionPass"> {
    let summary = "Shift data from a signed range to an unsigned one";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass detects quantized convolution and shifts weights data from a signed range to an unsigned one
    }];

    let constructor = "vpux::IE::createConvertWeightsToU8Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToMixPrecision
//

def ConvertToMixedPrecision: PassBase<"convert-to-mixed-precision", "vpux::FunctionPass"> {
    let summary = "Convert DPU task without fake quantize behind to mixed-precision operation";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.
        Converts DPU task to mixed-precision operation where there is no quantize operation for the output of a DPU task
    }];

    let constructor = "vpux::IE::createConvertToMixedPrecision()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertQuantizeOpsToNceOps
//

def ConvertQuantizeOpsToNceOps : PassBase<"convert-quantize-ops-to-nce-ops", "vpux::FunctionPass"> {
    let summary = "Converts per-tensor Quantize/Dequantize to eltwise And mixed-precision operation";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Converts per-tensor Quantize/Dequantize to eltwise And mixed-precision operation
        where input2 is input1 to perform type conversion on DPU instead of UPA.
    }];

    let constructor = "vpux::IE::createConvertQuantizeOpsToNceOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapQuantCastAndClamp
//

def SwapQuantCastAndClamp : PassBase<"swap-quant-cast-and-clamp", "vpux::FunctionPass"> {
    let summary = "Swap QuantizeCast and Clamp operations";

    let description = [{
        After AlignScales pass we have an additional Clamp layers in IR.
        Therefore, we may get such subgraph:
        ARG -> FQ -> Clamp -> Concat

        Then after SplitFakeQuant and PropagateQuantizeDequantize we have:
        ARG -> Q -> Clamp -> D -> Concat

        Then after ConvertQuantizeOpsToNceOps we have:
        ARG -> Add -> QuantizeCast -> Clamp -> QuantizeCast -> Add -> Concat

        In order to fuse Clamp into eltwise Add we need to move QuantizeCast after Clamp.
    }];

    let constructor = "vpux::IE::createSwapQuantCastAndClampPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseConvertWithQuantize
//

def FuseConvertWithQuantize : PassBase<"fuse-convert-with-quantize", "vpux::FunctionPass"> {
    let summary = "Fuse Convert with Quantize into QuantCast operation";

    let description = [{
        Pass detects pattern Convert(i8/ui8 -> FP16) -> Quantize(FP16 -> !quant.uniform<...>)
        and fuses it into single QuantCast(i8/ui8 -> !quant.uniform<...>) operation.
    }];

    let constructor = "vpux::IE::createFuseConvertWithQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertAvgPoolToDWConv
//

def ConvertAvgPoolToDWConv : PassBase<"convert-avg-pool-to-dw-conv", "vpux::FunctionPass"> {
    let summary = "Convert AvgPool op to GroupConvolution op";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces suitable `AvgPool` operations with `GroupConvolution` operation.
    }];

    let constructor = "vpux::IE::createConvertAvgPoolToDWConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleAsymmetricStrides
//

def HandleAsymmetricStrides : PassBase<"handle-asymmetric-strides", "vpux::FunctionPass"> {
    let summary = "Handle operations with asymmetric strides";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass splits operations so that they are able to be infered with symmetric strides
            on dpu because of hardware limitation.
    }];

    let constructor = "vpux::IE::createHandleAsymmetricStridesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePadOps
//

def FusePadOps : PassBase<"fuse-pad-ops", "vpux::FunctionPass"> {
    let summary = "Fuse PadOp with CONSTANT model";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        PadOp with CONSTANT model, pad value is 0 and the padding is needed in H and W dimensions only.
        Merge [Pad] -> [Conv] into [Conv].
        Merge [Pad] -> [GroupConv] into [GroupConv].
        Merge [Pad] -> [MaxPool] into [MaxPool].
    }];

    let constructor = "vpux::IE::createFusePadOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

}

//
// ConvertPadToConcat
//

def ConvertPadToConcat : PassBase<"convert-pad-to-concat", "vpux::FunctionPass"> {
    let summary = "Convert Pad Ops to Concat with Constant";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        After FusePadOps pass, there are Pad Ops can not be fused.
        Replace `IE::PadOp` with `IE::ConcatOp` and `Const::DeclareOp`
        Only `IE::PadMode::CONSTANT` case is supported.
    }];

    let constructor = "vpux::IE::createConvertPadToConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UpstreamSlice
//

def UpstreamSlice : PassBase<"upstream-slice", "vpux::FunctionPass"> {
    let summary = "Optimization by upstreaming slice operations";

    let description = [{
        Optimizes scenarios of IE::StridedSlice and IE::SliceOp without neighboring operations.
        Moves the slice operations upwards through the graph, reducing both compute and memory usage.
        In some cases the slice operation may be safely removed from the graph, if the action of upstreaming it
            only adapts the operations constants.
    }];

    let constructor = "vpux::IE::createUpstreamSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleLargeKernels
//

def HandleLargeKernels : PassBase<"handle-large-kernels", "vpux::FunctionPass"> {
    let summary = "Handle large kernels ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces average pooling layers that have kernels bigger than supported by hardware (11x11),
        with equivalent average pooling (approx equiv in case of prime kernel i.e. 13x13).
        For FP16-INT8: support to split into two average pooling.
        For FP16 / FP32: support multiple splits (converting to more than two average pooling ops) as there is no loss
                         of accuracy from the varying min/max ranges of the intermediate average pooling ops.
    }];

    let constructor = "vpux::IE::createHandleLargeKernelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertReduceToPooling
//

def ConvertReduceToPooling : PassBase<"convert-reduce-to-pooling", "vpux::FunctionPass"> {
    let summary = "Convert reduce to pooling ops";

    let description = [{
        The pass is to convert reduce operations (mean, max, sum, min) into pooling.
    }];

    let constructor = "vpux::IE::createConvertReduceToPoolingPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// InsertMaxpoolToConcatActivation
//

def InsertMaxpoolToConcatActivation : PassBase<"insert-maxpool-to-concat-activation", "vpux::FunctionPass"> {
    let summary = "Insert Maxpool op between Concat and Activation ops";

    let description = [{
        Pass converts Concat->Activation to Concat->Maxpool->Activation.
    }];

    let constructor = "vpux::IE::createInsertMaxpoolToConcatActivationPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleExcludePadForAvgPool
//

def HandleExcludePadForAvgPool : PassBase<"handle-exclude-pad-for-avg-pool", "vpux::FunctionPass"> {
    let summary = "Handle exclude-pad attribute for AvgPool operations";

    let description = [{
        This pass introduces exclude pad atribute handling for AvgPool operations, that have pad = stride = 1,
        by splitting operation in multiple AvgPool operations in order to handle this particular case.
    }];

    let constructor = "vpux::IE::createHandleExcludePadForAvgPoolPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// InsertReorderBetweenConcatAndLayers
//

def InsertReorderBetweenLayerAndConcat : PassBase<"layer-reorder-concat-pass", "vpux::FunctionPass"> {
    let summary = "Inserts Reorder operation between Transpose and Concat";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It inserts `Reorder` operation between layers `Transpose`, `AffineReshape` and `Concat` operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createInsertReorderBetweenLayerAndConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapTransposeWithFQ
//

def SwapTransposeWithFQ : PassBase<"swap-transpose-with-fq", "vpux::FunctionPass"> {
    let summary = "Swaps Transpose operation with FakeQuantize";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `Transpose` operation with per-tensor `FakeQuantize` operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createSwapTransposeWithFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapConvertWithTransposeReshape
//

def SwapConvertWithTransposeReshape : PassBase<"swap-convert-with-transpose-reshape", "vpux::FunctionPass"> {
    let summary = "Swaps Transpose operation with Convert";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `Transpose` and 'Reshape' operations with Convert operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createSwapConvertWithTransposeReshapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UniquifyBranches
//

def UniquifyBranches : PassBase<"uniquify-branches", "vpux::FunctionPass"> {
    let summary = "Eliminates redundant operations from multiple branches";

    let description = [{
        Convert this subgraph:
                 -> Slice -> Layer -> Consumer
        Producer -> Slice -> Layer -> Consumer
                 -> Slice -> Layer -> Consumer

        into this:
                          -> Slice -> Consumer
        Producer -> Layer -> Slice -> Consumer
                          -> Slice -> Consumer

        in case Slice and Layer transform different axes.

        Now at the place of "Layer" supported Reorder and Expand operations
    }];

    let constructor = "vpux::IE::createUniquifyBranchesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// TransposeToPermuteCast
//

def TransposeToPermuteCast : PassBase<"transpose-to-permute-cast", "vpux::FunctionPass"> {
    let summary = "Converts Transpose operation to PermuteCast with Reorder";

    let description = [{
        It is possible to replace a `Transpose` operation with a combination of `PermuteCast` and `Reorder`.
        To compute the permutation cast, which is required for the source tensor, one must inverse the
        affine map from the original `Transpose` operation. For example, consider the following transposition:
        `1x16x32x64 -> 1x64x16x32`, its affine map is: `(d0, d1, d2, d3) -> (d0, d3, d1, d2)`.
        The inverse will be:
        ```
            d0, d3, d1, d2   ->  d0, d1, d2, d3
            aN, aC, aH, aW   ->  aN, aH, aW, aC
        ```
        Which gives permutation cast into NHWC.
        In order to maintain the layout in data flow, `Reorder` must always rearrange `PermuteCast` result into the
        order of original `Transpose` operation.
    }];

    let constructor = "vpux::IE::createTransposeToPermuteCastPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateAffineReshape
//

def PropagateAffineReshape : PassBase<"propagate-affine-reshape", "vpux::FunctionPass"> {
    let summary = "Moves AffineReshape operation down";

    let description = [{
        Supported cases:
        * Move through Transpose
        * Move through Expand
        * Move through Concat // TODO: #-58713
            Before:
                AffineReshape ->
                AffineReshape -> Concat
                AffineReshape ->
            After:
                Concat -> AffineReshape
    }];

    let constructor = "vpux::IE::createPropagateAffineReshapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateFqThroughConcat
//

def PropagateFqThroughConcat : PassBase<"propagate-fq-through-concat", "vpux::FunctionPass"> {
    let summary = "Propagate FakeQuantize operation through Concat";

    let description = [{
        `ConvertPadToConcat` adds a `Concat` operation which does not propagate `FakeQuantize` operation.

        1. Check if such `Concat` operation has one and only one quantized input
        2. Fetch quantization parameters
        3. Apply them to every single `Concat` input and output
    }];

    let constructor = "vpux::IE::createPropagateFqThroughConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapConcatWithEltwise
//

def SwapConcatWithEltwise : PassBase<"swap-concat-with-eltwise", "vpux::FunctionPass"> {
    let summary = "Swaps Concat operation with elementwise -> FQ patterns";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `Concat` operation with elementwise -> FQ subgraph when possible. For example:
        * `PReLU` -> per-tensor `FakeQuantize` subgraph is eligible for such swap.

        This transormation allows to fuse `FakeQuantize` to NCE operations.
    }];

    let constructor = "vpux::IE::createSwapConcatWithEltwisePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PerAxisFQConcat
//

def PerAxisFQConcat : PassBase<"per-axis-fq-concat", "vpux::FunctionPass"> {
    let summary = "Supports Concat operation with per-axis FQ inputs";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It creates `FakeQuantize` operation, which combines per-channel quantization from `Concat` inputs,
        and places it after the `Concat` operation. For example:
        The following `Concat`:
        ```
            FQ 1x256x128x128 -> Concat <- FQ 1x48x128x128
                                  |
                                GroupConv 1x304x128x128
        ```
        will be transformed into:
        ```
            FQ 1x256x128x128 -> Concat <- FQ 1x48x128x128
                                  |
                                 FQ 1x304x128x128
                                  |
                                GroupConv 1x304x128x128
        ```
    }];

    let constructor = "vpux::IE::createPerAxisFQConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapTransposeConcat
//

def SwapTransposeConcat : PassBase<"swap-transpose-concat", "vpux::FunctionPass"> {
    let summary = "Swap Transpose and Concat operations";

    let description = [{
        Pass converts pattern from
        Transpose ->
        Transpose -> Concat
        Transpose ->

        to
        Concat -> Transpose
    }];

    let constructor = "vpux::IE::createSwapTransposeConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSquaredDiffToSubAndPower
//

def ConvertSquaredDiffToSubAndPower : PassBase<"convert-squareddiff-to-subpower", "vpux::FunctionPass"> {
    let summary = "Convert SquaredDifference operation to Subtract and Power";

    let description = [{
        This pass converts SquaredDifference operation to its equivalent (x-y)^2
    }];

    let constructor = "vpux::IE::createConvertSquaredDiffToSubAndPowerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPowerToMult
//

def ConvertPowerToMult : PassBase<"convert-power-to-mult", "vpux::FunctionPass"> {
    let summary = "Convert power to multiply operation";

    let description = [{
        The pass converts power with single constant exponent value to multiplication.
    }];

    let constructor = "vpux::IE::createConvertPowerToMultPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertUpsamplingToStridedConcat
//

def ConvertUpsamplingToStridedConcat : PassBase<"convert-upsampling-to-strided-concat", "vpux::FunctionPass"> {
    let summary = "Convert upsampling op to strided concat op";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Upsampling` operations with `Concat` operations with strides and a zero filled const.
    }];

    let constructor = "vpux::IE::createConvertUpsamplingToStridedConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeConcatSlice
//

def OptimizeConcatSlice : PassBase<"optimize-concat-slice", "vpux::FunctionPass"> {
    let summary = "Bypass concat if slice is the subtensor of one of concat inputs";

    let description = [{
        For the pattern ConcatOp->SliceOp, if SliceOp input is the subtensor of one of ConcatOp input,
        Bypass ConcatOp and ConcatOp would be removed if it has only one user.
    }];

    let constructor = "vpux::IE::createOptimizeConcatSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ExpandActivationWidth
//

def ExpandActivationWidth : PassBase<"expand-activation-width", "vpux::FunctionPass"> {
    let summary = "Align input tensors shape of DPU operation with hardware requirements";

    let description = [{
        This pass processes operations, which can be compiled as DPU tasks and
        expands output width to the next number divisible by 16 when they don't
        meet hardware requirements.
        Applicable only for operations with NHWC input and NCHW output.
        For instance, consider convolution with 16x20x23 input, 16x18x21 output and 3x3 kernel.
        21 is not divisible by 16, so the output width must be expanded to 32: 16x18x32.
        In order to comply to the operation traits, input width must be expanded to 16x20x34.

        Supported operations:
        * IE.Convolution
        * IE.GroupConvolution
        * IE.MaxPool
        * IE.AvgPool
        * IE.Add
    }];

    let constructor = "vpux::IE::createExpandActivationWidthPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::StandardOpsDialect"
    ];
}

//
// PropagateReorderToNCE
//

def PropagateReorderToNCE : PassBase<"propagate-reorder-to-nce", "vpux::FunctionPass"> {
    let summary = "Propagate reorder back to NCE task through act shave layers";

    let description = [{
        Converts these subgraphs:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
        ```
        Into the following subgraphs respectively:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
        ```
    }];

    let constructor = "vpux::IE::createPropagateReorderToNCEPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseReordersPass
//

def FuseReordersPass : PassBase<"fuse-reorders", "vpux::FunctionPass"> {
    let summary = "Fuses reorder to previous NCE task as ODU permutation";

    let description = [{
        Converts these subgraphs:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> IE.Reorder [NCHW]
        ```
        Into the following subgraphs respectively:
        ```
            Input [NHWC] -> IE.Convolution [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NCHW]
            Input [NHWC] -> IE.MaxPool [NCHW]
            Input [NHWC] -> IE.AvgPool [NCHW]
            Input [NHWC] -> IE.Add [NCHW]
        ```
    }];

    let constructor = "vpux::IE::createFuseReordersPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdaptShapesForScaleShiftPass
//

def AdaptShapesForScaleShiftPass : PassBase<"adapt-shapes-for-scale-shift", "vpux::FunctionPass"> {
    let summary = "Adjusts 2-d and 3-d `IE.Add` and `IE.Multiply` for further conversion to `IE.ScaleShift`";

    let description = [{
        Converts these subgraphs:
        ```
            Input tensor<NxM> => IE.Add : tensor<NxM>, tensor<1xM> -> tensor<NxM>
            Input tensor<NxM> => IE.Multiply : tensor<NxM>, tensor<1xM> -> tensor<NxM>
            Input tensor<1xNxM> => IE.Add : tensor<1xNxM>, tensor<1x1xM> -> tensor<1xNxM>
            Input tensor<1xNxM> => IE.Multiply : tensor<1xNxM>, tensor<1x1xM> -> tensor<1xNxM>
        ```
        Into the following subgraphs respectively:
        ```
            Input tensor<NxM> => IE.Add : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<NxM>
            Input tensor<NxM> => IE.Multiply : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<NxM>
            Input tensor<1xNxM> => IE.Add : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<1xNxM>
            Input tensor<1xNxM> => IE.Multiply : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<1xNxM>
        ```
        The following shape transformations will be applied for 2-d case:
        ```
            Input NxM => Reshape 1xNxMx1 => Transpose 1xMxNx1 => Add => Transpose 1xNxMx1 => Reshape NxM
        ```
        For 3-d case:
        ```
            Input 1xNxM => Reshape 1xNxMx1 => Transpose 1xMxNx1 => Add => Transpose 1xNxMx1 => Reshape 1xNxM
        ```
        It is also possible to apply reshape to get `IE.Add : tensor<NxMx1x1>, tensor<1xMx1x1>`.
        However, such approach may lead to a big cluster of NCE tasks after `UnrollBatch` pass.
        The measurements show that transposition is more effective for this pass.
    }];

    let constructor = "vpux::IE::createAdaptShapesForScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustInputShapeForEltwise
//

def AdjustInputShapeForEltwise : PassBase<"adjust-input-shape-for-eltwise", "vpux::FunctionPass"> {
    let summary = "Reshape the activation inputs of eltwise ops to make the inputs' channel aligned without `IE.Expand`";

    let description = [{
        Insert ShapeCast Ops to the inputs of eltwise ops when the input channels require alignment.
        Insert ShapeCast Ops to the outputs before the next non-eltwise op.
    }];

    let constructor = "vpux::IE::createAdjustInputShapeForEltwisePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MovePermutePostEltwise
//

def MovePermutePostEltwise : PassBase<"move-permute-post-eltwise", "vpux::FunctionPass"> {
    let summary = "Move the input Permute ops post Eltwise to reduce the number of Permute ops";

    let description = [{
        The layout does not matter for eltwise ops as long as the input and output layouts are the same.
        move the permute ops from the inputs of the eltwise to the output to reduce the number of permute ops.
    }];

    let constructor = "vpux::IE::createMovePermutePostEltwisePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertExtractImagePatches
//

def ConvertExtractImagePatches : PassBase<"convert-extract-image-patches", "vpux::FunctionPass"> {
    let summary = "Converts subgraphs around ExtractImagePatches into some more optimal for VPU ones when the necessary conditions are met";

    let description = [{
        Converts these subgraphs:
        ```
            IE.ReduceSum -> IE.ExtractImagePatches -> IE.Transpose -> IE.ReduceSum
            IE.ReduceSum -> IE.ExtractImagePatches -> IE.ReduceSum
            IE.ExtractImagePatches -> IE.Transpose -> IE.AffineReshape
            IE.ExtractImagePatches -> IE.Transpose
            IE.ExtractImagePatches
        ```
        Into the following subgraphs respectively:
        ```
            IE.ReduceSum -> IE.Unsqueeze
            IE.ReduceSum -> IE.Unsqueeze
            N x IE.Slice -> IE.Concat
            N x IE.Slice -> IE.Concat -> IE.AffineReshape
            IE.Transpose
        ```
    }];

    let constructor = "vpux::IE::createConvertExtractImagePatchesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// BroadcastInputForAdd
//

def BroadcastInputForAdd : PassBase<"broadcast-input-for-add", "vpux::FunctionPass"> {
    let summary = "Broadcast input for Add op";

    let description = [{
        This pass broadcast input for AddOp when the input1's shape isn't equal to input2's shape which cannot convert to ScaleShift.
    }];

    let constructor = "vpux::IE::createBroadcastInputForAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapMVNWithTranspose
//

def SwapMVNWithTranspose : PassBase<"swap-mvn-with-transpose", "vpux::FunctionPass"> {
    let summary = "Swaps MVN operation with parent Transpose";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `MVN` with Transpose operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createSwapMVNWithTransposePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


def AlignScales : PassBase<"align-scales", "vpux::FunctionPass"> {
    let summary = "Align FQ ranges around Concat layers.";

    let description = [{
        This pass aligns FQ ranges around Concat ops and inserts Clamp ops to keep the valuse in the original ranges.

        Original subgraph:
        Conv1                 ARG
          |                    |
        FQ1(range1)     FQ2(range2)
                      |
                    Concat
                      |
                    Conv2

        New subgraph:
        Conv1                                               ARG
          |                                                  |
        FQ(newRange)                                        FQ(newRange)
          |                                                  |
        Clamp(low and hight from original FQ)               Clamp(low and hight from original FQ)
                                                    |
                                                  Concat
                                                    |
                                                  Conv2
    }];


    let constructor = "vpux::IE::createAlignScalesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertReorderToPermuteQuantize
//

def ConvertReorderToPermuteQuantize : PassBase<"convert-reorder-to-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Converts IE.Reorder to DPU permute";

    let description = [{
        Converts IE.Reorder with float16 input and float16 output to DPU permute.
    }];

    let constructor = "vpux::IE::createConvertReorderToPermuteQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
