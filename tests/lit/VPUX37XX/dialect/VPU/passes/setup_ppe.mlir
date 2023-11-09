//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --setup-ppe %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882:128>

// CHECK-LABEL: @NoopCase
func.func @NoopCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 127 : i64,
            clamp_low = -128 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:         mode = <NOOP>,
    // CHECK-SAME:         clamp_low = 0 : i64,
    // CHECK-SAME:         clamp_high = 255 : i64,
    // CHECK-SAME:         lrelu_mult = 1 : i64,
    // CHECK-SAME:         lrelu_shift = 0 : i64,
    // CHECK-SAME:         quant_mult = [16822],
    // CHECK-SAME:         quant_shift = [13],
    // CHECK-SAME:         quant_post_shift = 0 : i64,
    // CHECK-SAME:         fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @ReLUCase
func.func @ReLUCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 2147483647 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:       ppe = #VPU.PPETask<
    // CHECK-SAME:           mode = <LRELU>,
    // CHECK-SAME:           clamp_low = 0 : i64,
    // CHECK-SAME:           clamp_high = 255 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64,
    // CHECK-SAME:           lrelu_shift = 0 : i64,
    // CHECK-SAME:           quant_mult = [16822],
    // CHECK-SAME:           quant_shift = [13],
    // CHECK-SAME:           quant_post_shift = 0 : i64,
    // CHECK-SAME:           fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:       >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @ReLUXCase
func.func @ReLUXCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 6 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:       ppe = #VPU.PPETask<
    // CHECK-SAME:           mode = <LRELUX>,
    // CHECK-SAME:           clamp_low = 0 : i64,
    // CHECK-SAME:           clamp_high = 6 : i64,
    // CHECK-SAME:           lrelu_mult = 1 : i64,
    // CHECK-SAME:           lrelu_shift = 0 : i64,
    // CHECK-SAME:           quant_mult = [16822],
    // CHECK-SAME:           quant_shift = [13],
    // CHECK-SAME:           quant_post_shift = 0 : i64,
    // CHECK-SAME:           fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:       >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @PReLUCase
func.func @PReLUCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 10 : i64,
            mode = <ADD>,
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        >
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 10 : i64,
    // CHECK-SAME:          quant_mult = [16822],
    // CHECK-SAME:          quant_shift = [13],
    // CHECK-SAME:          quant_post_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 9.765625E-4 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.061718750000000003:132>

// CHECK-LABEL: @MixedPrecision
func.func @MixedPrecision(%arg0: tensor<1x256x16x32xf16, {order =#NHWC}>,
                %arg1: tensor<1x256x16x32x!qElemType, {order = #NHWC}>) -> tensor<1x256x16x32x!qElemType, {order = #NHWC}> {
    %w = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<256x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<256x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

  
    %0 = VPU.NCE.DepthConvolution(%arg0, %w, %wt, %aw) {
        activation_window_channel_length = 4 : i64, 
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        rawFilterShape = [256, 1, 1, 1], 
        strides = [1, 1]
    } -> tensor<1x256x16x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution(%arg0, %cst, %cst_0, %cst_1) {
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 16.202531814575195 : f64
    // CHECK-SAME:      >

    return %0 : tensor<1x256x16x32x!qElemType, {order = #NHWC}>
}
