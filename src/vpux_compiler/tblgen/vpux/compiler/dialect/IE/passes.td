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

#ifndef VPUX_COMPILER_DIALECT_IE_PASSES
#define VPUX_COMPILER_DIALECT_IE_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// IECommon
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

//=================================================================================
// AdjustForVPU
//=================================================================================

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
// ConvertPrecisionToFP16
//

def ConvertPrecisionToFP16 : PassBase<"convert-precision-to-fp16", "vpux::ModulePass"> {
    let summary = "Convert tensors precision from FP32 to FP16";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

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
        The pass is a part of `AdjustForVPU` pipeline.
        This pass replaces all I64 tensors with I32.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToI32Pass()";

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
// ResolvePWLPostOps
//

def ResolvePWLPostOps : PassBase<"resolve-pwl-post-ops", "vpux::FunctionPass"> {
    let summary = "Resolve requirements for fused PWL post-ops";

    let description = [{
        Ensures the correct quantization ranges are used for fused PWL activation functions or
        unfuses them if surrounding tensors are not quantized per-tensor.
    }];

    let constructor = "vpux::IE::createResolvePWLPostOpsPass()";

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
        This pass is needed for MTL only since HW MaxPool does not support post-op operations.
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
    let summary = "Delete PerAxis Quantize Dequantize for MTL";

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
// ConvertQuantizeOpsToEltwise
//

def ConvertQuantizeOpsToEltwise : PassBase<"convert-quantize-ops-to-eltwise", "vpux::FunctionPass"> {
    let summary = "Converts per-tensor Quantize/Dequantize to eltwise And mixed-precision operation";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Converts per-tensor Quantize/Dequantize to eltwise And mixed-precision operation
        where input2 is input1 to perform type conversion on DPU instead of UPA.
    }];

    let constructor = "vpux::IE::createConvertQuantizeOpsToEltwisePass()";

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
        with equivalent two average pooling (approx equiv in case of prime kernel i.e. 13x13).
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
        The pass is to convert reduce operations (mean, max, sum) into pooling.
    }];

    let constructor = "vpux::IE::createConvertReduceToPoolingPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//=================================================================================
// Tiling
//=================================================================================

//
// Isolated Tiling
//

def IsolatedTiling : PassBase<"isolated-tiling", "vpux::FunctionPass"> {
    let summary = "Tile layers in isolation so that all their I/O fit into requirements";

    let description = [{
        The pass performs simple tiling of the layers, which doesn't fit into memory requirements.

        The pass tries to fit single layer in isolation, without any heuristics to run something
        in parallel or continue computation in tiles.

        The pass doesn't use any cost model and just choose the first largest tile size,
        which fits into the memory requirements.
    }];

    let constructor = "vpux::IE::createIsolatedTilingPass()";
}


def PrefetchTiling : PassBase<"prefetch-tiling", "vpux::FunctionPass"> {
    let summary = "Tile layers into smaller tiles to enable prefetch pipeline";

    let description = [{
        The pass performs tiling on layers to enable prefetch pipeline.

        The pass tries run tiles in parallel.
        The 'prefetch' means that the next tile could be loaded in advance when the current tile is computing.

        The pass does not consider cost models,
        only tiles layers to make at least two tiles could be loaded in CMX memory at the same time.
    }];

    let constructor = "vpux::IE::createPrefetchTilingPass()";
}

//
// InsertReorderBetweenConcatAndTranspose
//

def InsertReorderBetweenTransposeAndConcat : PassBase<"transpose-reorder-concat-pass", "vpux::FunctionPass"> {
    let summary = "Inserts Reorder operation between Transpose and Concat";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It inserts `Reorder` operation between `Transpose` and `Concat` operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createInsertReorderBetweenTransposeAndConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
