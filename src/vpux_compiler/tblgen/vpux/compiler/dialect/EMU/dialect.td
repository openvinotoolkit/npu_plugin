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

#ifndef VPUX_COMPILER_DIALECT_EMU
#define VPUX_COMPILER_DIALECT_EMU

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"

def EMU_Dialect : Dialect {
    let summary = "Emulator IR Dialect";

    let description = [{
The **EMU Dialect** represents the **IE Dialect** with additional logic to achieve compute equivalence.

It has the following properties:

* Maintains network representation close to original InferenceEngine one, to aid in debug scenarios.
* Describes network topology without HW details that don't affect compute (memory, sparsity, scheduling).
* Performs transformations to arrive at UPA/DPU operations at the same level as **VPUIP Dialect**.
* Facilitates graph serialization to graphfile schema inside MLIR framework.

[./EMU/_ops_interfaces.md]
    }];

    let name = "EMU";

    let cppNamespace = "vpux::EMU";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::IE::IEDialect",
        "mlir::StandardOpsDialect",
        "mlir::tensor::TensorDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let hasConstantMaterializer = 1;
}

#endif
