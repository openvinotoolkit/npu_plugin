//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvCannonicalize
func @DepthConvCannonicalize(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x1x1x4xsi32> =
        dense<1> : tensor<16x1x1x4xsi32>

    %0 = EMU.NCEClusterTask {
            kernel_padding = [0, 0, 0, 0],
            kernel_size = [4, 8],
            kernel_strides = [1, 1],
            rawFilterShape = [16, 1, 4, 8],
            task_type = "DWCONV"
        }
        input(%arg0 : tensor<1x16x40x80xf16, {order = #NHWC}>)
        weights(%cst0 : tensor<16x1x4x8xf16, {order = #NHWC}>)
        weight_table(%cst1 : tensor<16x1x1x4xsi32>)
        -> tensor<1x16x37x73xf16, {order = #NHWC}>
        PPE : {}

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x4x8xf16> = dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[16, 4, 8]>]
    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>


    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [0, 0, 0, 0], kernel_size = [4, 8], kernel_strides = [1, 1], rawFilterShape = [16, 1, 4, 8], task_type = "DWCONV"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x40x80xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights([[CST0]] : tensor<16x4x8xf16>)
    // CHECK-SAME:      weight_table([[CST]] : tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}
