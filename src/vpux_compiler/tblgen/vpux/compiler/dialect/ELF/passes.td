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

#ifndef VPUX_COMPILER_DIALECT_ELF_PASSES
#define VPUX_COMPILER_DIALECT_ELF_PASSES

include "mlir/Pass/PassBase.td"

/*
//
// SetCompileParams
//

def SetCompileParams2 : PassBase<"set-compile-params2", "vpux::ModulePass"> {
    let summary = "Set compilation parameters related to ELF Dialect";

    let description = [{
        This pass attaches compilation parameters related to **ELF Dialect** to Module attributes and
        initializes **IERT Dialect** run-time resources information.
    }];

    let constructor = "vpux::ELF::createSetCompileParamsPass2()";

    /*
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
    */

    let options = [
        Option<
            "testOpt", "test",
            "std::string", [{"KMB"}],
            "Test string"
        >
    ];

    let dependentDialects = [
        //"vpux::IERT::IERTDialect",
        "vpux::ELF::ELFDialect"
    ];
}
*/


/*
//
// AssignPhysicalBarriers
//

def AssignPhysicalBarriers : PassBase<"assign-physical-barriers", "vpux::FunctionPass"> {
    let summary = "Assign physical barriers";

    let description = [{
        This pass replaces virtual barriers with physical barriers and assign IDs to them.
    }];

    let constructor = "vpux::ELF::createAssignPhysicalBarriersPass()";

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

    let constructor = "vpux::ELF::createConvertWeightsTableOp2ConstPass()";
}

//
// DumpStatisticsOfTaskOps
//

def DumpStatisticsOfTaskOps : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of used Task operations";

    let description = [{
        This pass dumps the statistics of used Task operations and makes a report as warning for operations not converted to DPU.
    }];

    let constructor = "vpux::ELF::createDumpStatisticsOfTaskOpsPass()";
}
*/

#endif
