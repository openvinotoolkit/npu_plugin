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
// UPAProfiling
//

def UPAProfiling : PassBase<"upa-profiling", "vpux::ModulePass"> {
    let summary = "upa task profiling";

    let description = [{
        This pass allocate required memory in DDR space for UPA profiling and is own profiling output to the network
    }];

    let constructor = "vpux::VPUIP::createUPAProfilingPass()";

}

//
// GroupProfilingBuffers
//

def GroupProfilingBuffers : PassBase<"group-profiling-buffers", "vpux::ModulePass"> {
    let summary = "Group profiling buffers into single profiling output buffer";

    let description = [{
        Group profiling buffers from different profiling engines into single profiling output buffer with name as
        [offset]_[profiling name]_[offset]_[profiling name] so postprocessing tool can parse it back
    }];

    let constructor = "vpux::VPUIP::createGroupProfilingBuffersPass()";
}

//
// ConvertInstructionListTableOp2Const
//

def ConvertInstructionListTableOp2Const : PassBase<"convert-itable-op-to-constant", "vpux::FunctionPass"> {
    let summary = "Convert InstructionListTable Operations to IERT.ConstantOp";

    let description = [{
        This pass fills instruction list table.
    }];

    let constructor = "vpux::VPUIP::createConvertInstructionListTableOp2ConstPass()";
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
        1. Find VPUIP::NNDMAOp with Const::DeclareOp source and VPURT::DeclareBufferOp target.
        2. Check that weights size matches minimal compression size.
        3. Compress weights.
        4. Wrap compressed weights to flat tensor shapes with UInt8 data type.
        5. Replace original VPUIP::NNDMAOp with VPUIP::CompressedDMAOp
    }];

    let constructor = "vpux::VPUIP::createCompressWeightsPass()";
}

//
// UnrollClusterTiling
//

def UnrollClusterTiling : PassBase<"unroll-cluster-tiling", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from ClusterTiling operation";

    let description = [{
        Legalizes chain of ClusterTiling operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::createUnrollClusterTilingPass()";
}

#endif
