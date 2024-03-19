//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// UnrollClusterTiling
//

def UnrollClusterTiling : PassBase<"unroll-cluster-tiling", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from ClusterTiling operation";

    let description = [{
        Legalizes chain of ClusterTiling operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::arch37xx::createUnrollClusterTilingPass()";
}

//
// AddSwKernelCacheHandlingOps
//

def AddSwKernelCacheHandlingOps : PassBase<"add-sw-kernel-cache-handling-ops", "vpux::FunctionPass"> {
    let summary = "Add cache-handling ops for SW Kernels";

    let description = [{
        This pass adds a CACHE_FLUSH_INVALIDATE SwKernel after each SwKernel that has at least
        one input or output buffers located in DDR such that the ActSHAVE cache is both flushed
        and invalidated right after an ActSHAVE Kernel reads/writes to DDR, through the cache.
    }];

    let constructor = "vpux::VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass()";
}

#endif
