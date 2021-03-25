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
