//
// Copyright Intel Corporation.
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
// ConvertIE2IERT
//

def ConvertIE2IERT : PassBase<"convert-IE-to-IERT", "vpux::FunctionPass"> {
    let summary = "Bufferize the IE dialect into the IERT dialect on Function level";

    let description = [{
        This pass bufferizes IE Dialect into IERT Dialect.
        Only functions inner regions are affected, not signatures.
    }];

    let constructor = "vpux::createConvertIE2IERTPass()";

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect"
    ];
}

//
// ConvertIERT2VPUIP
//

def ConvertIERT2VPUIP : PassBase<"convert-IERT-to-VPUIP", "vpux::FunctionPass"> {
    let summary = "Convert IERT Dialect Layer Operations to VPUIP Dialect";

    let description = [{
        This pass converts Laye Operations from IERT Dialect to VPUIP Dialect.
        It uses simple SW implementation for IERT operations without any optimizations.
    }];

    let constructor = "vpux::createConvertIERT2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
