//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_CONVERSION_PASSES
#define VPUX_COMPILER_CONVERSION_PASSES

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

    let constructor = "vpux::createConvertIEToVPUNCEPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertIEToVPUM2I
//

def ConvertIEToVPUM2I : PassBase<"convert-IE-to-VPU-M2I", "vpux::FunctionPass"> {
    let summary = "Convert supported IE layers to VPU M2I operations";

    let constructor = "vpux::createConvertIEToVPUM2IPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertLayers2VPU
//

def ConvertLayers2VPU : PassBase<"convert-layers-to-VPU", "vpux::FunctionPass"> {
    let summary = "Convert Layers Operations to VPU Dialect (SW and View layers)";

    let constructor = "vpux::createConvertLayers2VPUPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::IE::IEDialect"
    ];
}

//=================================================================================
// LowerIE2IERT
//=================================================================================

//
// BufferizeIE
//

def BufferizeIE : PassBase<"bufferize-IE", "vpux::FunctionPass"> {
    let summary = "Bufferize the IE dialect into the IERT dialect on Function level";

    let description = [{
        This pass bufferizes **IE Dialect** into **IERT Dialect**:

        * Updates only Function inner regions.
        * Doesn't change Function signatures.
        * Replaces Layer Operations with IERT analogues.
        * Replaces Quant Dialect Operations with IERT analogues.
    }];

    let constructor = "vpux::createBufferizeIEPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "mlir::StandardOpsDialect",
        "mlir::memref::MemRefDialect"
    ];
}

//
// BufferizeFuncAndReturn
//

def BufferizeFuncAndReturn : PassBase<"bufferize-func-and-return", "vpux::ModulePass"> {
    let summary = "Bufferize func/return ops";

    let description = [{
        A bufferize pass that bufferizes std.func ops.
        In contrast to MLIR standard analogue pass, this pass uses vpux::BufferizeTypeConverter
        to process encoding attribute in mlir::RankedTensorType
    }];

    let constructor = "vpux::createBufferizeFuncAndReturnPass()";
}

//
// AddBuffersForNetResults
//

def AddBuffersForNetResults : PassBase<"add-buffers-for-net-results", "vpux::ModulePass"> {
    let summary = "Add network results in Function parameters";

    let description = [{
        This pass adds buffers to the function parameters to copy the network result to them.
        In contrast to MLIR standard analogue pass, this pass preserves the function results to utilize use-def chains on bufferized IR.
        The `return` operation will take an aliases of output buffers from the function arguments.
    }];

    let constructor = "vpux::createAddBuffersForNetResults()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "mlir::StandardOpsDialect",
        "mlir::memref::MemRefDialect"
    ];
}

//=================================================================================
// Lowering VPU to VPUIP
//=================================================================================

//
// ConvertVPUNCEToVPUIP
//

def ConvertVPUNCEToVPUIP : PassBase<"convert-vpu-nce-to-vpuip", "vpux::FunctionPass"> {
    let summary = "Convert NCE ops from VPU to VPUIP";

    let description = [{
        Convert ops which can be executed on NCE to VPUIP::NCEClusterTask ops.
    }];

    let constructor = "vpux::createConvertVPUNCEToVPUIPPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertNCEClusterTilingToVPUIP
//

def ConvertNCEClusterTilingToVPUIP : PassBase<"convert-nce-cluster-tiling-to-vpuip", "vpux::FunctionPass"> {
    let summary = "Convert VPU NCEClusterTiling ops to VPUIP";

    let description = [{
        Convert VPU::NCEClusterTiling ops to VPUIP::NCEClusterTiling.
    }];

    let constructor = "vpux::createConvertNCEClusterTilingToVPUIPPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertLayers2VPUIP
//

def ConvertLayers2VPUIP : PassBase<"convert-layers-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert VPU Operations to VPUIP Dialect";

    let description = [{
        Convert software and view-like operations from VPU to VPUIP dialect.
        Software operations are converted to UPA tasks, supported by VPUX30XX.
    }];

    let constructor = "vpux::createConvertLayers2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertSWLayers2VPUIP
//

def ConvertSWLayers2VPUIP : PassBase<"convert-sw-layers-to-VPUIP", "vpux::ModulePass"> {
    let summary = "Convert Software Operations to VPUIP SWKernel";

    let description = [{
        Convert software operations from VPU to VPUIP::SWKernel operations for VPU37XX.
    }];

    let constructor = "vpux::createConvertSWLayers2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "mlir::arith::ArithmeticDialect",
        "vpux::VPURT::VPURTDialect"
    ];

}


//
// ConvertVPUIP2VPUIPRegMapped
//

def ConvertVPUIP2VPUIPRegMapped : PassBase<"convert-VPUIP-to-VPUIPRegMapped", "vpux::ModulePass"> {
    let summary = "Convert VPUIP Dialect to VPUIPRegMapped Dialect";

    let constructor = "vpux::createConvertVPUIP2VPUIPRegMappedPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUIPRegMapped::VPUIPRegMappedDialect",
        "vpux::ELF::ELFDialect"
    ];
}


//
// Convert2VPUIPRegMappedAndELF
//

def Convert2VPUIPRegMappedAndELF : PassBase<"convert-to-VPUIPRegMapped-and-ELF", "vpux::ModulePass"> {
    let summary = "Convert to VPUIPRegMapped Dialect and ELF Dialect";

    let constructor = "vpux::createConvert2VPUIPRegMappedAndELFPass()";

    let dependentDialects = [
        "vpux::VPURT::VPURTDialect", // E#47129
        "vpux::VPUIPRegMapped::VPUIPRegMappedDialect",
        "vpux::ELF::ELFDialect"
    ];
}

#endif
