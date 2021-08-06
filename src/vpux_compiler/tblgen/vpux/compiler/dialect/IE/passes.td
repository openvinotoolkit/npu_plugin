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

//
// UseUserPrecision
//

def UseUserPrecision : PassBase<"use-user-precision", "vpux::ModulePass"> {
    let summary = "Use user precisions for entry point function prototype";

    let description = [{
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
// ConvertShapeTo4D
//

def ConvertShapeTo4D : PassBase<"convert-shape-to-4d", "vpux::ModulePass"> {
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

        It replaces IE::StridedSlice operation with simple StridedSlice and Reshape.
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
// PropagateQuantizeDequantize
//

def PropagateQuantizeDequantize : PassBase<"propagate-quantize-dequantize", "vpux::FunctionPass"> {
    let summary = "Update quantize/dequantize ops";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass detect pattern `quant.dcast -> op -> quant.qcast` into single quantized Op
    }];

    let constructor = "vpux::IE::createPropagateQuantizeDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

#endif
