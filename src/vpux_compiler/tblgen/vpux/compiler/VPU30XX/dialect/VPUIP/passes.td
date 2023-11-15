//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ARCH_30XX_PASSES
#define VPUX_COMPILER_DIALECT_VPUIP_ARCH_30XX_PASSES

include "mlir/Pass/PassBase.td"

//
// UnrollClusterTiling
//

def UnrollClusterTiling : PassBase<"unroll-cluster-tiling", "vpux::FunctionPass"> {
    let summary = "Extracts child operations from ClusterTiling operation";

    let description = [{
        Legalizes chain of ClusterTiling operations according to the tile schema
    }];

    let constructor = "vpux::VPUIP::arch30xx::createUnrollClusterTilingPass()";
}

#endif
