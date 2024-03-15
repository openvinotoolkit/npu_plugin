//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
        "vpux::VPUIP::VPUIPDialect",
        "mlir::func::FuncDialect",
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
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect"
    ];
}

//=================================================================================
// Lowering VPU to VPUIP
//=================================================================================

//
// OneShotBufferizeVPU2VPUIP
//

def OneShotBufferizeVPU2VPUIP : PassBase<"one-shot-bufferize-VPU-to-VPUIP", "vpux::ModulePass"> {
    let summary = "One shot bufferize ops from VPU to VPUIP";

    let description = [{
        One shot bufferize ops from VPU to VPUIP dialect, include converting:
        (1) software ops from VPU to VPUIP::SWKernel ops. (37XX+)
        (2) software ops from VPU to VPUIP::XXXUPAOp ops. (30XX)
        (3) ops which can be executed on NCE to VPUIP::NCEClusterTask ops. (30XX+)
        (4) VPU::NCEClusterTiling ops to VPUIP::NCEClusterTiling. (30XX+)
    }];

    let constructor = "vpux::createOneShotBufferizeVPU2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "mlir::memref::MemRefDialect",
        "mlir::bufferization::BufferizationDialect"
    ];
}

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
        Convert operations from VPU to VPUIP dialect,
        including remaining operations except for NCE and SW operations.
    }];

    let constructor = "vpux::createConvertLayers2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertSWLayers2VPUIPSWKernel
//

def ConvertSWLayers2VPUIPSWKernel : PassBase<"convert-sw-layers-to-VPUIP-sw-kernel", "vpux::ModulePass"> {
    let summary = "Convert Software Operations to VPUIP SWKernel";

    let description = [{
        Convert software operations from VPU to VPUIP::SWKernel operations.
    }];

    let constructor = "vpux::createConvertSWLayers2VPUIPSWKernelPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "mlir::arith::ArithDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertSWLayers2VPUIPUPA
//

def ConvertSWLayers2VPUIPUPA : PassBase<"convert-sw-layers-to-VPUIP-UPA", "vpux::FunctionPass"> {
    let summary = "Convert Software Operations to VPUIP UPAOp";

    let description = [{
        Convert software operations from VPU to VPUIP::XXXUPAOp operations.
    }];

    let constructor = "vpux::createConvertSWLayers2VPUIPUPAPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect"
    ];
}

//
// ConvertSWLayers2Affine
//

def ConvertSWLayers2Affine : PassBase<"convert-sw-layers-to-Affine", "vpux::ModulePass"> {
    let summary = "Convert Software Operations to Affine Dialect";

    let description = [{
        Convert software operations from IERT to Affine operations for VPU37XX.
    }];

    let constructor = "vpux::createConvertSWLayers2AffinePass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::affine::AffineDialect",
        "mlir::memref::MemRefDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::scf::SCFDialect",
        "mlir::math::MathDialect"
    ];
}


//
// ConvertAffine2LLVM
//

def ConvertAffine2LLVM : PassBase<"convert-Affine-to-LLVM", "vpux::ModulePass"> {
    let summary = "Convert Affine and Std dialects to LLVM dialect";

    let constructor = "vpux::createConvertAffine2LLVMPass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::affine::AffineDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::memref::MemRefDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::scf::SCFDialect",
        "mlir::math::MathDialect"
    ];
}

//
// ConvertVPUIP2VPUMI37XX
//

def ConvertVPUIP2VPUMI37XX : PassBase<"convert-VPUIP-to-VPUMI37XX", "vpux::ModulePass"> {
    let summary = "Convert VPUIP Dialect to VPUMI37XX Dialect";

    let constructor = "vpux::createConvertVPUIP2VPUMI37XXPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPURT::VPURTDialect",
        "vpux::VPUMI37XX::VPUMI37XXDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

//
// ConvertVPUMI37XX2ELF
//

def ConvertVPUMI37XX2ELF : PassBase<"convert-VPUMI37XX-to-ELF", "vpux::ModulePass"> {
    let summary = "Convert VPUMI37XX Dialect to ELF Dialect";

    let constructor = "vpux::createConvertVPUMI37XX2ELFPass()";

    let dependentDialects = [
        "vpux::VPURT::VPURTDialect", // E#47129
        "vpux::VPUMI37XX::VPUMI37XXDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect"
    ];
}

#endif
