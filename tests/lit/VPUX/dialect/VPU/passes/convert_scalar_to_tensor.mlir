//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scalar-to-tensor %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Gather
func @Gather(%arg0: tensor<18x8x72x64xf16>) -> tensor<8x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si32>

    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64}
            : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<8x72x64xf16>

    return %0 : tensor<8x72x64xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si32>
    // CHECK:       [[VAL0:%.*]] = VPU.Reshape([[CST]]) {shape_value = [1], special_zero} : tensor<si32> -> tensor<1xsi32>
    // CHECK:       [[VAL1:%.*]] = VPU.Gather(%arg0, [[VAL0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<18x8x72x64xf16>, tensor<1xsi32> -> tensor<1x8x72x64xf16>
    // CHECK:       [[VAL2:%.*]] = VPU.Reshape([[VAL1]]) {shape_value = [8, 72, 64]} : tensor<1x8x72x64xf16> -> tensor<8x72x64xf16>
    // CHECK:       return [[VAL2]]
}

// CHECK-LABEL: @TopK
func @TopK(%arg0: tensor<6x12x10x24xf16>) -> (tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>) {
    %cst = const.Declare tensor<si32> = dense<3> : tensor<si32>

    %0:2 = VPU.TopK(%arg0, %cst) {axis = 1 : i64, mode = "MAX", sort = "SORT_VALUES", element_type = i64}
            : tensor<6x12x10x24xf16>, tensor<si32> -> tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>

    return %0#0, %0#1 : tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<si32> = dense<3> : tensor<si32>
    // CHECK:       [[VAL0:%.*]] = VPU.Reshape([[CST]]) {shape_value = [1], special_zero} : tensor<si32> -> tensor<1xsi32>
    // CHECK:       [[VAL1:%.*]], [[VAL2:%.*]] = VPU.TopK(%arg0, [[VAL0]])
    // CHECK-SAME:      {axis = 1 : i64, element_type = i64, mode = "MAX", sort = "SORT_VALUES"}
    // CHECK-SAME:      : tensor<6x12x10x24xf16>, tensor<1xsi32> -> tensor<6x3x10x24xf16>, tensor<6x3x10x24xi64>
    // CHECK:       return [[VAL1]], [[VAL2]]
}
