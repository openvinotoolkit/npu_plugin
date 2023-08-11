//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-vpu-nce-to-emu %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE
func.func @ConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x1x1x4xsi32> =
        dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
            pad = {bottom = 0, left = 0, right = 0, top = 0},
            rawFilterShape = [16, 16, 1, 1],
            ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "LRELU"},
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [0, 0, 0, 0], kernel_size = [1, 1], kernel_strides = [1, 1], rawFilterShape = [16, 16, 1, 1], task_type = "CONV"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x16x16xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights([[CST0]] : tensor<16x16x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:      weight_table([[CST]] : tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCE
func.func @DepthConvToNCE(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<16x1x1x4xsi32> =
        dense<1> : tensor<16x1x1x4xsi32>
    %cst2 = const.Declare tensor<1x1x1x16xui8> =
        dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %cst1, %cst2) {
            pad = {bottom = 0, left = 0, right = 0, top = 0},
            rawFilterShape = [16, 1, 4, 8],
            strides = [1, 1],
            activation_window_channel_length = 44
        } -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [0, 0, 0, 0], kernel_size = [4, 8], kernel_strides = [1, 1], rawFilterShape = [16, 1, 4, 8], task_type = "DWCONV"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x40x80xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights([[CST0]] : tensor<16x1x4x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      weight_table([[CST]] : tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCE
func.func @MaxPoolToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x1x4xsi32> =
        dense<1> : tensor<16x1x1x4xsi32>
    %cst1 = const.Declare tensor<1x1x1x16xui8> =
        dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %cst0, %cst1) {
            kernel_size = [1, 1],
            pad = {bottom = 0, left = 0, right = 0, top = 0},
            strides = [1, 1],
            activation_window_channel_length = 4
        } -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [0, 0, 0, 0], kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "MAXPOOL"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x1x4xf16, {order = #NHWC}>)
    // CHECK-SAME:      weight_table([[CST0]] : tensor<16x1x1x4xsi32>)
    // CHECK-SAME:      ->  tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCE
func.func @AvgPoolToNCE(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            kernel_size = [3, 3],
            pad = {bottom = 1, left = 1, right = 1, top = 1},
            strides = [1, 1],
            ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_mult = [28835], quant_shift = [18]}
        } -> tensor<1x16x4x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [1, 1, 1, 1], kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "AVEPOOL"
    // CHECK-SAME:      input(%arg0 : tensor<1x16x4x4xf16, {order = #NHWC}>)
    // CHECK-SAME:      ->  tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK:       PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [28835], quant_shift = [18]}

    // CHECK:       return [[VAL0]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCE
func.func @EltwiseAddToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647, clamp_low = 0 , lrelu_mult = 1, lrelu_shift = 0, mode = "ADD"}} :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights(%arg1 : tensor<1x64x28x28xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK:           PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}

    // CHECK:       return [[VAL0]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAndSameInputsToNCE
func.func @EltwiseAndSameInputsToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "AND",
        ppe = {clamp_high = 2147483647, clamp_low = 0 , lrelu_mult = 1, lrelu_shift = 0, mode = "AND"}} :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

     // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK:           PPETask "AND" {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}

    // CHECK:       return [[VAL0]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}
