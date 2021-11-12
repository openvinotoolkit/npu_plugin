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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_PASSES

include "mlir/Pass/PassBase.td"

//
// SetCompileParams
//

def SetCompileParams : PassBase<"set-compile-params", "vpux::ModulePass"> {
    let summary = "Set compilation parameters related to VPUIP Dialect";

    let description = [{
        This pass attaches compilation parameters related to **VPUIP Dialect** to Module attributes and
        initializes **IERT Dialect** run-time resources information.
    }];

    let constructor = "vpux::VPUIP::createSetCompileParamsPass()";

    let options = [
        Option<
            "archOpt", "vpu-arch",
            "std::string", [{"KMB"}],
            "VPU architecture to compile for"
        >,
        Option<
            "compilationModeOpt", "compilation-mode",
            "std::string", [{"DefaultHW"}],
            "Set compilation mode as reference (ReferenceSW) or hardware (ReferenceHW or DefaultHW)"
        >,
        Option<
            "numberOfDPUGroupsOpt", "num-of-dpu-groups",
            "int", "",
            "Number of DPU groups"
        >
    ];

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "vpux::VPUIP::VPUIPDialect"
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

    let constructor = "vpux::VPUIP::createAssignPhysicalBarriersPass()";

    let options = [
        Option<
            "_numBarriersOpt", "num-barriers",
            "int", "",
            "Number of physical barriers, available for usage"
        >
    ];
}

//
// BarrierSimulation
//

def BarrierSimulation : PassBase<"barrier-simulation", "vpux::FunctionPass"> {
    let summary = "Simulate barriers";

    let description = [{
        Simulates the physical barrier execution and stops compilation on fail.
    }];

    let constructor = "vpux::VPUIP::createBarrierSimulationPass()";
}

//
// ConvertWeightsTableOp2Const
//

def ConvertWeightsTableOp2Const : PassBase<"convert-wtable-op-to-constant", "vpux::FunctionPass"> {
    let summary = "Convert WeightsTable Operations to IERT.ConstantOp";

    let description = [{
        This pass fills weights table considering the information about the offset in the memory of the weights or activation window.
    }];

    let constructor = "vpux::VPUIP::createConvertWeightsTableOp2ConstPass()";
}

//
// DumpStatisticsOfTaskOps
//

def DumpStatisticsOfTaskOps : PassBase<"dump-statistics-of-task-ops", "vpux::FunctionPass"> {
    let summary = "Dump the statistics of used Task operations";

    let description = [{
        This pass dumps the statistics of used Task operations and makes a report as warning for operations not converted to DPU.
    }];

    let constructor = "vpux::VPUIP::createDumpStatisticsOfTaskOpsPass()";
}

//
// CompressWeights
//

def CompressWeights : PassBase<"compress-weights", "vpux::FunctionPass"> {
    let summary = "Compress binary data when possible";

    let description = [{
        This pass applies bitcompactor to tensor binary data. The logic is the following:
        1. Find VPUIP::NNDMAOp with Const::DeclareOp source and VPUIP::DeclareTensorOp target.
        2. Check that weights size matches minimal compression size.
        3. Compress weights.
        4. Wrap compressed weights to flat tensor shapes with UInt8 data type.
        5. Replace original VPUIP::NNDMAOp with VPUIP::CompressedDMAOp
    }];

    let constructor = "vpux::VPUIP::createCompressWeightsPass()";
}

#endif
