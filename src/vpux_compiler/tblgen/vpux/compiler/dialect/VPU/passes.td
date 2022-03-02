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

#ifndef VPUX_COMPILER_DIALECT_VPU_PASSES
#define VPUX_COMPILER_DIALECT_VPU_PASSES

include "mlir/Pass/PassBase.td"

//
// InitCompiler
//

def InitCompiler : PassBase<"init-compiler", "vpux::ModulePass"> {
    let summary = "Initializes compiler for VPU platforms";

    let description = [{
        This pass attaches VPU related compilation parameters to Module attributes and
        initializes **IERT Dialect** run-time resources information.
    }];

    let constructor = "vpux::VPU::createInitCompilerPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", [{"KMB"}],
            "VPU architecture to compile for"
        >,
        Option<
            "compilationModeOpt", "compilation-mode",
            "std::string", [{"DefaultHW"}],
            "Set compilation mode as `ReferenceSW`, `ReferenceHW` or `DefaultHW`"
        >,
        Option<
            "numberOfDPUGroupsOpt", "num-of-dpu-groups",
            "int", "",
            "[Optional] Number of available DPU groups"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "vpux::VPU::VPUDialect"
    ];
}

//
// Wrap multi-cluster layers in NCEClusterTiling
//

def WrapVPUOpsInNCEClusterTiling : PassBase<"wrap-vpu-ops-in-ncecluster-tiling", "vpux::FunctionPass"> {
    let summary = "This pass wraps vpu operations that should be executed across multiple clusters in NCEClusterTiling operations";

    let description = [{
        This pass builds an IR in order to represent multi-cluster compilation. It performs a number of functions.
        1) It creates variations of distributed tensors depending on the multi-cluster strategy of the layer. 
        2) It creates DMA operations DDR->CMX and wraps the DMAs in NCEClusterTiling.
        3) It wraps hardware executable operations in NCEClusterTiling. 
    }];

    let constructor = "vpux::VPU::createWrapVPUOpsInNCEClusterTilingPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitNCEOpsOntoWorkloads
//

def SplitNCEOpsOntoWorkloads : PassBase<"split-NCE-ops-onto-workloads", "vpux::FunctionPass"> {
    let summary = "Split VPU NCE operation onto workloads";

    let constructor = "vpux::VPU::createSplitNCEOpsOntoWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
