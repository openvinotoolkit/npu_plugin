//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --break-data-flow %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @LinearGraph
func.func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %0 = memref.alloc() : memref<10xf16>
    %1 = memref.alloc() : memref<10xf16>
    %token, %results = async.execute -> !async.value<memref<10xf16>> attributes {"async-deps-index" = 0 : i64} {
      %3 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%0 : memref<10xf16>) -> memref<10xf16>
      async.yield %3 : memref<10xf16>
    }
    %token_0, %results_1 = async.execute [%token] (%results as %arg2: !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> attributes {"async-deps-index" = 1 : i64} {
      %3 = IERT.ReLU inputs(%arg2 : memref<10xf16>) outputs(%1 : memref<10xf16>) -> memref<10xf16>
      async.yield %3 : memref<10xf16>
    }
    %2 = async.await %results_1 : !async.value<memref<10xf16>>
    return %2 : memref<10xf16>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<10xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:       IERT.ReLU
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16>)
    // CHECK-NEXT:  async.yield [[BUF0]] : memref<10xf16>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%arg[0-9]]]: !async.value<memref<10xf16>>)
    // CHECK:       IERT.ReLU
    // CHECK-SAME:      inputs([[VAL1]] : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<10xf16>)
    // CHECK-NEXT:  async.yield [[BUF1]] : memref<10xf16>

    // CHECK:       [[VAL3:%.+]] = async.await [[F2]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL3]]
}
