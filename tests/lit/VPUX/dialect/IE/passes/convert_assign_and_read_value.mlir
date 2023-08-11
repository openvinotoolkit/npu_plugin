//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-assign-read-value %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertAssignAndReadValue
IE.CNNNetwork entryPoint : @ConvertAssignAndReadValue inputsInfo : {
    DataInfo "input1" : tensor<1x768xf32>
} outputsInfo : {
    DataInfo "Gemm_9" : tensor<1x768xf32>
}
func.func @ConvertAssignAndReadValue(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {
    %cst = const.Declare tensor<1x768xf32> = dense<1.100000e+00> : tensor<1x768xf32>
    %0 = IE.ReadValue(%cst) {name = "inner_h1"} : tensor<1x768xf32> -> tensor<1x768xf32>
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<1x768xf32> -> tensor<1x768xf32>
    %2 = IE.Assign(%1) {name = "inner_h1"} : tensor<1x768xf32> -> tensor<1x768xf32>
    return %1 : tensor<1x768xf32>

    // CHECK-NOT:   IE.ReadValue
    // CHECK-NOT:   IE.Assign
    // CHECK:       DataInfo "vpux_ie_read_value_inner_h1" : tensor<1x768xf32>
    // CHECK:       DataInfo "vpux_ie_assign_inner_h1" : tensor<1x768xf32>
    // CHECK:       @ConvertAssignAndReadValue(%arg0: tensor<1x768xf32>, %arg1: tensor<1x768xf32>) -> (tensor<1x768xf32>, tensor<1x768xf32>)
    // CHECK:       [[VAR0:%.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<1x768xf32> -> tensor<1x768xf32>
    // CHECK:       return [[VAR0]], [[VAR0]] : tensor<1x768xf32>, tensor<1x768xf32>
}
