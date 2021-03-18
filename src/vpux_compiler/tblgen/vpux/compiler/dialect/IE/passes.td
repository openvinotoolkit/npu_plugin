//
// Copyright 2020 Intel Corporation.
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
// ConvertPrecisionToFP16
//

def ConvertPrecisionToFP16 : Pass<"convert-precision-to-fp16", "mlir::ModuleOp"> {
    let summary = "Convert tensors precision from FP32 to FP16";

    let description = [{
        This "atomic" pass replaces all FP32 tensors with FP16.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToFP16Pass()";
}

//
// ConvertShapeTo4D
//

def ConvertShapeTo4D : Pass<"convert-shape-to-4d", "mlir::ModuleOp"> {
    let summary = "Convert tensors shapes to 4D";

    let description = [{
        This "atomic" pass replaces all ND tensor with 4D analogues.
        Tensor with rank greater than 4 will be treated as an error.
    }];

    let constructor = "vpux::IE::createConvertShapeTo4DPass()";

    let dependentDialects = [
        "mlir::linalg::LinalgDialect"
    ];
}

//
// ConvertPaddingsToFloorMode
//

def ConvertPaddingsToFloorMode : FunctionPass<"convert-paddings-to-floor-mode"> {
    let summary = "Convert Convolution and Pooling layers paddings to FLOOR rouding mode";

    let description = [{
        This "atomic" pass updates padding attributes for Convolution and Pooling layers.
        It switches layer rounding mode to FLOOR and updates paddings to satisfy output shape.
    }];

    let constructor = "vpux::IE::createConvertPaddingsToFloorModePass()";
}

//
// ConvertTile2PerAxisTile
//

def ConvertTile2PerAxisTile : FunctionPass<"convert-tile-to-per-axis-tiles"> {
    let summary = "Convert tile op by multiple axes to multiple PerAxisTile ops";

    let description = [{
        This "atomic" pass replaces all Tile op with PerAxisTile ops.
    }];

    let constructor = "vpux::IE::createConvertTile2PerAxisTilePass()";
}

//
// AdjustForVPU
//

def AdjustForVPU : Pass<"adjust-for-vpu", "mlir::ModuleOp"> {
    let summary = "Adjust IE Dialect IR for VPU target";

    let description = [{
        This "pipeline" pass includes various adaptation passes to adjust the IR for VPU target.
    }];

    let constructor = "vpux::IE::createAdjustForVPUPass()";

    let dependentDialects = [
        "mlir::linalg::LinalgDialect"
    ];
}

//
// UseUserPrecision
//

def UseUserPrecision : Pass<"use-user-precision", "mlir::ModuleOp"> {
    let summary = "Use user precisions for entry point function prototype";

    let description = [{
        This "atomic" pass updates the CNNNetwork entry point function prototype
        and use user-provided precisions for its operands and results.
        The pass inserts Convert operations from/to topology precisions.
    }];

    let constructor = "vpux::IE::createUseUserPrecisionPass()";
}

//
// SplitFakeQuant
//

def SplitFakeQuant : FunctionPass<"split-fake-quant"> {
    let summary = "Splits IE FakeQuantize operations to `quant.qcast -> quant.dcast` pair";

    let constructor = "vpux::IE::createSplitFakeQuantPass()";

    let dependentDialects = [
        "mlir::quant::QuantizationDialect"
    ];
}

//
// QuantizeConst
//

def QuantizeConst : FunctionPass<"quantize-const"> {
    let summary = "Performs constant folding for `Constant -> quant.qcast` case";

    let constructor = "vpux::IE::createQuantizeConstPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DequantizeConst
//

def DequantizeConst : FunctionPass<"dequantize-const"> {
    let summary = "Performs constant folding for `Constant -> quant.dcast` case";

    let constructor = "vpux::IE::createDequantizeConstPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MergeFakeQuant
//

def MergeFakeQuant : FunctionPass<"merge-fake-quant"> {
    let summary = "Merge pair `quant.qcast -> quant.dcast` into single IE.FakeQuantize";

    let constructor = "vpux::IE::createMergeFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LowPrecision
//

def LowPrecision : FunctionPass<"low-precision"> {
    let summary = "Low precision transformations";

    let description = [{
        This "pipeline" pass includes all transformations to support low precisions.
    }];

    let constructor = "vpux::IE::createLowPrecisionPass()";

    let dependentDialects = [
        "mlir::quant::QuantizationDialect",
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertFCToConv
//

def ConvertFCToConv : FunctionPass<"convert-fc-to-conv"> {
    let summary = "Convert FullyConnected op to Convolution op";

    let description = [{
        This "atomic" pass replaces all FullyConnected ops with Convolution ops.
    }];

    let constructor = "vpux::IE::createConvertFCToConvPass()";
}

#endif
