//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --add-weights-table-to-eltwise %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AddWeightsTableToEltwiseOps
func @AddWeightsTableToEltwiseOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = EMU.NCEClusterTask {
            task_type = "ELTWISE"
        }
        input(%arg0 : tensor<1x16x16x16xf16, {order = #NHWC}>)
        weights(%arg1 : tensor<1x16x16x16xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
        PPE : {
            PPETask "ADD" {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0}
        }

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<{{.*}}> : tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights(%arg1 : tensor<1x16x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:      weight_table([[CST0]] : tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AddWeightsTableToAvgPoolOps
func @AddWeightsTableToAvgPoolOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = EMU.NCEClusterTask {
            task_type = "AVEPOOL",
            kernel_padding = [1, 1, 1, 1],
            kernel_size = [3, 3],
            kernel_strides = [1, 1]
        }
        input(%arg0 : tensor<1x16x16x16xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
        PPE : {
            PPETask "NOOP" {clamp_high = 2147483647, clamp_low = -2147483648, lrelu_mult = 1, lrelu_shift = 0, quant_mult = [28835], quant_shift = [18]}
        }

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<{{.*}}> : tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [1, 1, 1, 1],
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      kernel_strides = [1, 1],
    // CHECK-SAME:      task_type = "AVEPOOL"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:      weight_table([[CST0]] : tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [28835], quant_shift = [18]}

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}
