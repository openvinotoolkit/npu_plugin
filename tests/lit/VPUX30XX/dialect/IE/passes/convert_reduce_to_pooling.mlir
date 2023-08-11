//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s

// CHECK-LABEL: @NotConvertReduceMean
func.func @NotConvertReduceMean(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<[3]> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
  return %0 : tensor<1x32x112x1xf16>

  // CHECK:       %0 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x32x112x112xf16>, tensor<1xsi32> -> tensor<1x32x112x1xf16>
  // CHECK-NOT:   AvgPool
}
