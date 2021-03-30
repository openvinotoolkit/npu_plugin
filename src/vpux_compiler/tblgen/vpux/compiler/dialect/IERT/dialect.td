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
The **IERT Dialect** represents bufferized version of **IE Dialect**.

It has the following properties:

* Works with fixed operation set (like **IE Dialect**).
* Represents execution scheduling and memory allocation.
* Works with `MemRefType`.
* Includes transformations and optimizations closer to HW level (memory re-usage, parallel resources usage, etc.).

**TBD:** It operates with `MemRefType`, but in contrast to MLIR uses SSA value semantic (inspired by PlaidML approach).
It combines both memory effects and buffer aliasing for this:

* Each layer operation takes as its operands both input and output buffers.
* The layer marks input buffer as read-only and output buffer as write-only via memory effects inferface.
* The layer returns new buffer Value, which is an alias for output buffer.

```MLIR
#NHWC = affine_map<(n, c, h, w) -> (n, h, w, c)>

func @main(%input: memref<1x3x240x240xf16, #NHWC>, %output: memref<1x3x240x240xf16, #NHWC>) -> memref<1x3x240x240xf16, #NHWC> {
    %1 = IERT.SoftMax(%input, %output) {axisInd = 1 : i32} // %1 is an alias for %output
    return %1
}
```

The memory allocation/deallocation is defined as separate operations (dynamic or static).

The **IERT Dialect** uses the following scheme to represent scheduling information:

* Operations order defines scheduling.
* Each IERT operation is assumed as blocking: next operation will not start until previous is finished.
* Concurrent execution is defined as asynchronous regions (**Async Dialect**).

```MLIR
%11_t, %11_f = async.execute { IERT.executor = "NCE_Cluster" }
    [%7_t, %8_9_t](%8_9_f#0 as %8, %8_9_f#1 as %9)
{
    %11_0_t, %11_0_f = async.execute { IERT.executor = "DPU" }
    {
        %11_0 = IERT.Convolution(%7, %8, %9) to %10_0 { strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1] }
        async.yield %11_0
    }

    %11_1_t, %11_1_f = async.execute { IERT.executor = "DPU" }
    {
        %11_1 = IERT.Convolution(%7, %8, %9) to %10_1 { strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1] }
        async.yield %11_1
    }

    %11:2 = async.await %11_0_f, %11_1_f
    %11 = IERT.FakeConcat(%11#0, %11#1) to %10
    async.yield %11
}
```

The **IERT Dialect** provides separate Operation to describe the available and used run-time resources.
It deals with the following resource types:

* Memory space.
* Executor (CPU, HW module, DMA).

```MLIR
IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1073741824 bytes
        IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 4194304 bytes of "CMX_UPA" {VPUIP.bandwidth = 16 : i64, VPUIP.derateFactor = 8.500000e-01 : f64}
        IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory : {
        IERT.MemoryResource 2048 bytes of "DDR"
        IERT.MemoryResource 1048576 bytes of "CMX_NN"
    }
    executors : {
        IERT.ExecutorResource 1 of "Leon_RT"
        IERT.ExecutorResource 1 of "Leon_NN"
        IERT.ExecutorResource 16 of "SHAVE_UPA"
        IERT.ExecutorResource 20 of "SHAVE_NN"
        IERT.ExecutorResource 4 of "NCE_Cluster" {
            IERT.ExecutorResource 5 of "NCE_PerClusterDPU"
        }
        IERT.ExecutorResource 1 of "DMA_UPA"
        IERT.ExecutorResource 1 of "DMA_NN"
    }
```

The `IERT.RunTimeResources` is filled by underlying low-level dialect to provide information about HW-specific resources.
    }];

    let name = "IERT";

    let cppNamespace = "vpux::IERT";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::StandardOpsDialect",
        "mlir::memref::MemRefDialect",
        "mlir::quant::QuantizationDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

#endif
