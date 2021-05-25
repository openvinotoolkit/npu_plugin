//
// Copyright 2020 Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_IE
#define VPUX_COMPILER_DIALECT_IE

include "mlir/IR/OpBase.td"

def IE_Dialect : Dialect {
    let summary = "InferenceEngine IR Dialect";

    let description = [{
The **IE Dialect** represents InferenceEngine/nGraph IR in terms of MLIR framework.

It has the following properties:

* Describes network topology without HW details (memory hierarchy, memory allocation, scheduling).
* Represents the latest nGraph opset and in addition some portion of legacy IE opset (for convenience).
* Works with MLIR Tensor Types as atomic Values (no memory effects), all operations are pure.
* Performs high level transformations/optimizations, that doesn't need low level details (memory buffers, layouts, scheduling).

Some of the layer operations in the **IE Dialect** defines Canonicalization hooks to simplify IR for further optimizations:

* Remove redundant Operations (same type `Reshape`/`Tile`, `Add` with 0, etc.).
* Apply Lazy Constant Folding.
* Replace Constant Values with Attributes (more linear graph).
* Fuse common patterns (`Mul+Add => ScaleShift`, `Convolution+Bias`).
* Use more convinient Operations (`MatMul => FullyConnected`, `Reshape => linalg.tensor_reshape`).

Quantization parameters are stored as a part of tensor/buffer element type (`QuantizedType` from **Quant Dialect**).

The network topology (nGraph) is represented as a MLIR Function, which works with `tensor` types.

```MLIR
func @main(%input: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %output = IE.SoftMax(%input) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %output
}
```

The network inputs and outputs information (names, precision, layout) is held in separate Operation - `IE.CNNNetwork`.

```MLIR
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x3x400x400xf32>
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x1000xf32>
    }
```
    }];

    let name = "IE";

    let cppNamespace = "vpux::IE";

    let dependentDialects = [
        "mlir::StandardOpsDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let hasConstantMaterializer = 1;
}

#endif
