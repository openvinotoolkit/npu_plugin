//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// RegMappedand your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_PASSES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_PASSES

include "mlir/Pass/PassBase.td"


//
// SetCompileParams
//

def SetCompileParams : PassBase<"set-compile-params", "vpux::ModulePass"> {
    let summary = "Set compilation parameters related to VPUIPRegMapped Dialect";

    let description = [{
        This pass attaches compilation parameters related to **VPUIPRegMapped Dialect** to Module attributes and
        initializes **IERT Dialect** run-time resources information.
    }];

    let constructor = "vpux::VPUIPRegMapped::createSetCompileParamsPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", [{"KMB"}],
            "VPU architecture to compile for"
        >,
        Option<
            "compilationModeOpt", "compilation-mode",
            "std::string", [{"ReferenceSW"}],
            "Set compilation mode as reference (ReferenceSW) or hardware (ReferenceHW)"
        >,
        Option<
            "numberOfDPUGroupsOpt", "num-of-dpu-groups",
            "int", "",
            "Number of DPU groups"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "vpux::VPUIPRegMapped::VPUIPRegMappedDialect" // TODO small check - might have to comment it
    ];
}

//
// AssignPhysicalBarriers
//

def AssignPhysicalBarriers : PassBase<"assign-physical-barriers", "vpux::FunctionPass"> {
    let summary = "Assign physical barriers";

    let description = [{
        This pass replaces virtual barriers with physical barriers and assign IDs to them.
    }];

    let constructor = "vpux::VPUIPRegMapped::createAssignPhysicalBarriersPass()";

    let options = [
        Option<
            "_numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for usage"
        >
    ];
}

//
// ConvertWeightsTableOp2Const
//

def ConvertWeightsTableOp2Const : PassBase<"convert-wtable-op-to-constant", "vpux::FunctionPass"> {
    let summary = "Convert WeightsTable Operations to IERT.ConstantOp";

    let description = [{
        This pass fills weights table considering the information about the offset in the memory of the weights or activation window.
    }];

    let constructor = "vpux::VPUIPRegMapped::createConvertWeightsTableOp2ConstPass()";
}

//
// DumpStatisticsOfTaskOps
//

def DumpStatisticsOfTaskOps : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of used Task operations";

    let description = [{
        This pass dumps the statistics of used Task operations and makes a report as warning for operations not converted to DPU.
    }];

    let constructor = "vpux::VPUIPRegMapped::createDumpStatisticsOfTaskOpsPass()";
}

#endif
