//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_CONVERSION_ARCH_37XX_PASSES
#define VPUX_COMPILER_CONVERSION_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// LowerIE2VPU
//=================================================================================

//
// ConvertIEToVPUNCE
//

def ConvertIEToVPUNCE : PassBase<"convert-IE-to-VPU-NCE", "vpux::FunctionPass"> {
    let summary = "Convert supported IE layers to VPU NCE operations";

    let description = [{
        The pass converts VPUNCE-friendly operations (such as Conv, Pool, Eltwise)

        It doesn't change the memory spaces or add any new operations such as Copy
    }];

    let constructor = "vpux::arch37xx::createConvertIEToVPUNCEPass()";

    let options = [
        Option<
            "useNCEPermute", "use-nce-permute",
            "bool", "false",
            "Use NCE Permute operation"
        >
    ];

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertLayers2VPU
//

def ConvertLayers2VPU : PassBase<"convert-layers-to-VPU", "vpux::FunctionPass"> {
    let summary = "Convert Layers Operations to VPU Dialect (SW and View layers)";

    let constructor = "vpux::arch37xx::createConvertLayers2VPUPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::IE::IEDialect"
    ];
}

#endif
