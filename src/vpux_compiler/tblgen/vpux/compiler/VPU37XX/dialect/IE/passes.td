//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

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

#endif
