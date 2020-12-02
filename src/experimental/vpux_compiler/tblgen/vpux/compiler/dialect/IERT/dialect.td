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

#ifndef VPUX_COMPILER_DIALECT_IERT
#define VPUX_COMPILER_DIALECT_IERT

include "mlir/IR/OpBase.td"

def IERT_Dialect : Dialect {
    let summary = "InferenceEngine RunTime Dialect";

    let description = [{
        The `IERT` Dialect represents InferenceEngine/nGraph IR closer to run-time.

        It operates with MLIR MemRef Types, introduces initial allocation and scheduling logic.
    }];

    let name = "IERT";

    let cppNamespace = "vpux::IERT";

    let dependentDialects = [
        "mlir::StandardOpsDialect"
    ];

    let hasConstantMaterializer = 1;
}

#endif
