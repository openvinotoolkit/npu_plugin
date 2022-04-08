//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
