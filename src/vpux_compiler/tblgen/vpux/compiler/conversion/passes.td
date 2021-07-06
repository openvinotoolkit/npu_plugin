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

#ifndef VPUX_COMPILER_CONVERSION_PASSES
#define VPUX_COMPILER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

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
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect",
        "mlir::memref::MemRefDialect"
    ];
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
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect",
        "mlir::memref::MemRefDialect"
    ];
}

//=================================================================================
// Lowering to NCE
//=================================================================================

//
// ConvertToNCEOps
//

def ConvertToNCEOps : PassBase<"convert-to-nce-ops", "vpux::FunctionPass"> {
    let summary = "Convert to NCE2 ops";

    let description = [{
        Convert ops which can be executed on NCE to explicit NCE ops.
    }];

    let constructor = "vpux::createConvertToNCEOpsPass()";

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//=================================================================================
// LowerIERT2VPUIP
//=================================================================================

//
// ConvertLayers2VPUIP
//

def ConvertLayers2VPUIP : PassBase<"convert-layers-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert Layers Operations to VPUIP Dialect (UPA and DMA tasks)";

    let constructor = "vpux::createConvertLayers2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertDeclarations2VPUIP
//

def ConvertDeclarations2VPUIP : PassBase<"convert-declarations-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert declarations (constants and memory buffers) to VPUIP Dialect";

    let constructor = "vpux::createConvertDeclarations2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertViewOps2VPUIP
//

def ConvertViewOps2VPUIP : PassBase<"convert-view-ops-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert view-like Operations to VPUIP Dialect";

    let constructor = "vpux::createConvertViewOps2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// ConvertAsyncOps2VPUIP
//

def ConvertAsyncOps2VPUIP : PassBase<"convert-async-ops-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert Async Dialect Operations to VPUIP Dialect";

    let description = [{
        This pass inlines 'async.execute' body to parent Block and replaces '!async.token' based dependencies with
        VPUIP virtual barriers.
    }];

    let constructor = "vpux::createConvertAsyncOps2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
