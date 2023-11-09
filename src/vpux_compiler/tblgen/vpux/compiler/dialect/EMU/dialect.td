//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect",
        "mlir::func::FuncDialect",
        "mlir::tensor::TensorDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let hasConstantMaterializer = 1;

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif
