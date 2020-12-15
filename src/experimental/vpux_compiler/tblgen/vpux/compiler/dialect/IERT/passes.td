//
// Copyright 2020 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_IERT_PASSES
#define VPUX_COMPILER_DIALECT_IERT_PASSES

include "mlir/Pass/PassBase.td"

//
// StaticAllocation
//

def StaticAllocation : Pass<"static-allocation", "mlir::ModuleOp"> {
    let summary = "Replace dynamic allocations with static";

    let description = [{
        This "atomic" pass replaces all dynamic `alloc`/`dealloc` Operations with `IERT.StaticAlloc`.
        It uses simple LinearScan algorithm.
    }];

    let constructor = "vpux::IERT::createStaticAllocationPass(nullptr)";

    let options = [
        Option<
            "memSpaceName", "memory-space",
            "std::string", [{""}],
            "Memory space to perform allocation"
        >
    ];
}

#endif
