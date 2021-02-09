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

#ifndef VPUX_COMPILER_DIALECT_VPUIP
#define VPUX_COMPILER_DIALECT_VPUIP

include "mlir/IR/OpBase.td"

def VPUIP_Dialect : Dialect {
    let summary = "VPU NN RunTime Dialect";

    let description = [{
        The `VPUIP` Dialect represents NN RunTime IR in terms of MLIR framework.
    }];

    let name = "VPUIP";

    let cppNamespace = "vpux::VPUIP";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

#endif
