//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef VPUX_COMPILER_DIALECT_IE_PASSES
#define VPUX_COMPILER_DIALECT_IE_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Dialect/Linalg/IR/LinalgOps.td"

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
}

//
// ConvertShapeTo4D
//

def ConvertShapeTo4D : PassBase<"convert-shape-to-4d", "vpux::ModulePass"> {
    let summary = "Convert tensors shapes to 4D";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        **TBD:** This pass replaces ND tensor with 4D analogues for layers, which has such limitations on VPUIP level.
    }];

    let constructor = "vpux::IE::createConvertShapeTo4DPass()";

    let dependentDialects = [
        "mlir::linalg::LinalgDialect"
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
        "mlir::quant::QuantizationDialect"
    ];
}

//
// QuantizeConst
//

def QuantizeConst : PassBase<"quantize-const", "vpux::FunctionPass"> {
    let summary = "Quantize constant tensors";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It performs constant folding for `Constant -> quant.qcast` case.
    }];

    let constructor = "vpux::IE::createQuantizeConstPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
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

#endif
