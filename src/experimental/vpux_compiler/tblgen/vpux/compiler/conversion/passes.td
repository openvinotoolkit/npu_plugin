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
// ConvertIE2Quant
//

def ConvertIE2Quant : FunctionPass<"convert-IE-to-quant"> {
    let summary = "Converts IE FakeQuantize operations to analogues from quant Dialect";

    let constructor = "vpux::createConvertIE2QuantPass()";

    let dependentDialects = [
        "mlir::quant::QuantizationDialect"
    ];
}

//
// BufferizeIE
//

def BufferizeIE : FunctionPass<"bufferize-IE"> {
    let summary = "Bufferize the IE dialect into the IERT dialect on Function level";

    let description = [{
        This "atomic" pass bufferizes IE Dialect into IERT Dialect.
        Only functions inner regions are affected, not signatures.
    }];

    let constructor = "vpux::createBufferizeIEPass()";

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect"
    ];
}

//
// LowerIE2IERT
//

def LowerIE2IERT : Pass<"lower-IE-to-IERT", "mlir::ModuleOp"> {
    let summary = "Performs full lowering from the IE Dialect to IERT Dialect";

    let description = [{
        This "pipeline" pass performs full IR lowering from IE Dialect to IERT Dialect,
        including Function types, call graph and return operations.
    }];

    let constructor = "vpux::createLowerIE2IERTPass()";

    let dependentDialects = [
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

//
// ConvertIERT2VPUIP
//

def ConvertIERT2VPUIP : FunctionPass<"convert-IERT-to-VPUIP"> {
    let summary = "Convert IERT Dialect Layer Operations to VPUIP Dialect";

    let description = [{
        This "atomic" pass converts Laye Operations from IERT Dialect to VPUIP Dialect.
        It uses simple SW implementation for IERT operations without any optimizations.
    }];

    let constructor = "vpux::createConvertIERT2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

//
// LowerIERT2VPUIP
//

def LowerIERT2VPUIP : Pass<"lower-IERT-to-VPUIP", "mlir::ModuleOp"> {
    let summary = "Performs full lowering from IERT Dialect to VPUIP Dialect";

    let description = [{
        This "pipeline" pass performs full IR lowering from IERT Dialect to VPUIP Dialect,
        including Module level Operations.
    }];

    let constructor = "vpux::createLowerIERT2VPUIPPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
