//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-i32 --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @GatherConvertIndices
func.func @GatherConvertIndices(%arg0: tensor<100xf16>) -> tensor<10xf16> {
  %0 = const.Declare tensor<10xsi64> = dense<1> : tensor<10xsi64>

  %prob = IE.Gather(%arg0, %0) {axis_value = 0, batch_dims = 0} : tensor<100xf16>,tensor<10xsi64> -> tensor<10xf16>

  return %prob : tensor<10xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<10xsi32> = dense<1> : tensor<10xsi64>, [#const.ConvertElemType<si32>]
  //CHECK: [[VAL1:%.*]] = IE.Gather(%arg0, [[VAL0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<100xf16>, tensor<10xsi32> -> tensor<10xf16>
  //CHECK: return [[VAL1]]
}

// CHECK-LABEL: @EqualConvert
func.func @EqualConvert(%arg0: tensor<1x10x1xsi64>) -> tensor<1x10x1xsi64> {
  %0 = const.Declare tensor<1x1x1xsi64> = dense<0> : tensor<1x1x1xsi64>
  %1 = IE.Equal(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1xsi64>, tensor<1x1x1xsi64> -> tensor<1x10x1xsi64>
  return %1 : tensor<1x10x1xsi64>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<1x1x1xsi32> = dense<0> : tensor<1x1x1xsi64>, [#const.ConvertElemType<si32>]
  //CHECK: [[VAL1:%.*]] = IE.Equal(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1xsi32>, tensor<1x1x1xsi32> -> tensor<1x10x1xsi32>
  //CHECK: return [[VAL1]]
}


