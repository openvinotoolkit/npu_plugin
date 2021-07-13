// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --group-async-execute-ops %s | FileCheck %s

// CHECK-LABEL: @MergeUPAAndDMA
func @MergeUPAAndDMA(%arg0: memref<1x3x1x1xui8>, %arg1: memref<1x3x1x1xf16>, %arg2: memref<1x3x1x1xf16>) -> (memref<1x3x1x1xf16>, memref<1x3x1x1xf16>) {
  %token, %results = async.execute -> !async.value<memref<1x3x1x1xf16, "DDR">> attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
    %2 = IERT.StaticAlloc<0> -> memref<1x3x1x1xf16, "DDR">
    %3 = IERT.Convert inputs(%arg0 : memref<1x3x1x1xui8>) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
    async.yield %3 : memref<1x3x1x1xf16, "DDR">
  }
  async.await %token : !async.token
  %token_0, %results_1 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>) -> !async.value<memref<1x3x1x1xf16, "DDR">> attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
    %2 = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
    %3 = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
    async.yield %3 : memref<1x3x1x1xf16, "DDR">
  }
  async.await %token_0 : !async.token
  %token_2, %results_3 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>) -> !async.value<memref<1x3x1x1xf16, "DDR">> attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
    %2 = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
    %3 = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
    async.yield %3 : memref<1x3x1x1xf16, "DDR">
  }
  async.await %token_2 : !async.token
  %token_4, %results_5 = async.execute (%results_3 as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>) -> !async.value<memref<1x3x1x1xf16>> attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
    %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg1 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
    async.yield %2 : memref<1x3x1x1xf16>
  }
  %0 = async.await %results_5 : !async.value<memref<1x3x1x1xf16>>
  %token_6, %results_7 = async.execute (%results_1 as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>) -> !async.value<memref<1x3x1x1xf16>> attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
    %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg2 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
    async.yield %2 : memref<1x3x1x1xf16>
  }
  %1 = async.await %results_7 : !async.value<memref<1x3x1x1xf16>>
  return %0, %1 : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>
}

// CHECK:         [[T:%.+]], [[F:%.+]] = async.execute
// CHECK:         async.await [[T]]

// CHECK:         [[T0:%.+]], [[F1:%.+]] = async.execute
// CHECK:                 [[VAR0:%.*]] = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
// CHECK:                 [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64}
// CHECK-SAME:                            inputs([[ARG3:%.*]] : memref<1x3x1x1xf16, "DDR">) outputs([[VAR0]] : memref<1x3x1x1xf16, "DDR">)
// CHECK-SAME:                            -> memref<1x3x1x1xf16, "DDR">
// CHECK:                 [[VAR2:%.*]] = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
// CHECK:                 [[VAR3:%.*]] = IERT.SoftMax {axisInd = 1 : i64}
// CHECK-SAME:                            inputs([[ARG4:%.*]] : memref<1x3x1x1xf16, "DDR">) outputs([[VAR2]] : memref<1x3x1x1xf16, "DDR">)
// CHECK-SAME:                            -> memref<1x3x1x1xf16, "DDR">
// CHECK:                 async.yield [[VAR1]], [[VAR3]] : memref<1x3x1x1xf16, "DDR">, memref<1x3x1x1xf16, "DDR">
// CHECK:         async.await [[T0]]

// CHECK:         [[T1:%.+]], [[F2:%.+]]:2  = async.execute
// CHECK:                 [[VAR5:%.*]] = IERT.Copy
// CHECK-SAME:                            inputs([[ARG5:%.*]] : memref<1x3x1x1xf16, "DDR">) outputs([[ARG1:%.*]] : memref<1x3x1x1xf16>)
// CHECK-SAME:                            -> memref<1x3x1x1xf16>
// CHECK:                 [[VAR7:%.*]] = IERT.Copy
// CHECK-SAME:                            inputs([[ARG6:%.*]] : memref<1x3x1x1xf16, "DDR">) outputs([[ARG2:%.*]] : memref<1x3x1x1xf16>)
// CHECK-SAME:                            -> memref<1x3x1x1xf16>
// CHECK:                 async.yield [[VAR5]], [[VAR7]] : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>
// CHECK:         [[VAR8:%.*]] = async.await [[F2]]#0
// CHECK:         [[VAR9:%.*]] = async.await [[F2]]#1

// CHECK:       return [[VAR8]], [[VAR9]] : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>

// CHECK-LABEL: @MergeDMAs
func @MergeDMAs(%arg0: memref<1x3x1x1xui8>, %arg1: memref<1x3x1x1xf16>, %arg2: memref<1x3x1x1xf16>) -> (memref<1x3x1x1xf16>, memref<1x3x1x1xf16>) {
  %token, %results = async.execute -> !async.value<memref<1x3x1x1xf16, "DDR">> attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
    %2 = IERT.StaticAlloc<0> -> memref<1x3x1x1xf16, "DDR">
    %3 = IERT.Convert inputs(%arg0 : memref<1x3x1x1xui8>) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
    async.yield %3 : memref<1x3x1x1xf16, "DDR">
  }
  async.await %token : !async.token

  %token_4, %results_5 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>) -> !async.value<memref<1x3x1x1xf16>> attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
    %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg1 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
    async.yield %2 : memref<1x3x1x1xf16>
  }
  %0 = async.await %results_5 : !async.value<memref<1x3x1x1xf16>>
  %token_6, %results_7 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>) -> !async.value<memref<1x3x1x1xf16>> attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
    %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg2 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
    async.yield %2 : memref<1x3x1x1xf16>
  }
  %1 = async.await %results_7 : !async.value<memref<1x3x1x1xf16>>
  return %0, %1 : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>
}

// CHECK:         [[T:%.+]], [[F:%.+]] = async.execute
// CHECK:         async.await [[T]]

// CHECK:         [[T1:%.+]], [[F2:%.+]]:2 = async.execute
// CHECK:                 [[VAR1:%.*]] = IERT.Copy
// CHECK-SAME:                            inputs([[ARG1:%.*]] : memref<1x3x1x1xf16, "DDR">) outputs([[ARG2:%.*]] : memref<1x3x1x1xf16>)
// CHECK-SAME:                            -> memref<1x3x1x1xf16>
// CHECK:                 [[VAR2:%.*]] = IERT.Copy
// CHECK-SAME:                            inputs([[ARG3:%.*]] : memref<1x3x1x1xf16, "DDR">) outputs([[ARG4:%.*]] : memref<1x3x1x1xf16>)
// CHECK-SAME:                            -> memref<1x3x1x1xf16>
// CHECK:                 async.yield [[VAR1]], [[VAR2]] : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>
// CHECK:         [[VAR4:%.*]] = async.await [[F2]]#0
// CHECK:         [[VAR5:%.*]] = async.await [[F2]]#1

// CHECK:         return [[VAR4]], [[VAR5]] : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>