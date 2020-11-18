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

#ifndef VPUX_COMPILER_CONVERSION_PASSES
#define VPUX_COMPILER_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//
// ConvertIE2VPUIP
//

def ConvertIE2VPUIP : Pass<"convert-IE-to-VPUIP", "mlir::ModuleOp"> {
    let summary = "Convert the operations from the IE dialect into the VPUIP dialect";

    let description = [{
        This conversion uses simple SW implementation for IE operations without any optimizations.
    }];

    let constructor = "vpux::createConvertIE2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect",
        "mlir::StandardOpsDialect",
        "mlir::linalg::LinalgDialect"
    ];

    let options = [
        Option<
            "maxUPAShaves",
            "max-UPA-SHAVEs",
            "uint32_t",
            /*default=*/"1",
            "Maximal number of UPA SHAVEs to utilize"
        >
    ];
}

#endif
