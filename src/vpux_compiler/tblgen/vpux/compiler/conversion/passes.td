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
        * Replaces `linalg.tensor_reshape` with `linalg.reshape`.
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
// ConvertToNCEOps
//=================================================================================

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
// ConvertIERTOps2VPUIP
//

def ConvertIERTOps2VPUIP : PassBase<"convert-IERT-ops-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert IERT Dialect Operations to VPUIP Dialect";

    let description = [{
        This pass converts IERT Layer Operations to VPUIP Dialect UPA tasks.
        It also converts IERT declarations and view-like operations to VPUIP declarations.
    }];

    let constructor = "vpux::createConvertIERTOps2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// FuseActivations
//

def FuseActivations : PassBase<"fuse-activations", "vpux::FunctionPass"> {
    let summary = "Fuse activation functions to NCE tasks";

    let description = [{
        Fuse activation functions (e.g. ReLU, leaky ReLU) into NCE post-processing engine.
    }];

    let constructor = "vpux::createFuseActivationsPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
