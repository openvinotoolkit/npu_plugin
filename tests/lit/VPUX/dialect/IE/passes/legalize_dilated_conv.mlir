//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-dilated-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @LegalizeDilatedConvolution
func @LegalizeDilatedConvolution(%arg0: tensor<1x2x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x2x16x16xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x2x16x16xf32>

    %weights_low = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
    %weights_high = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %weights = const.Declare tensor<8x2x3x3xf32> = dense<5.0> : tensor<8x2x3x3xf32>
    %1 = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high)
        { auto_broadcast = "NUMPY", levels = 255 } :
        tensor<8x2x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<8x2x3x3xf32>

    %2 = IE.Convolution(%0, %1)
        {
            strides = [1, 1],
            pads_begin = [2, 2],
            pads_end = [2, 2],
            dilations = [2, 2]
        } :
        tensor<1x2x16x16xf32>, tensor<8x2x3x3xf32> -> tensor<1x8x16x16xf32>

    return %2 : tensor<1x8x16x16xf32>

    // CHECK: [[MIN_IN:%.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK: [[MAX_IN:%.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK: [[FQ0:%.*]] = IE.FakeQuantize(%arg0, [[MIN_IN]], [[MAX_IN]], [[MIN_IN]], [[MAX_IN]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x2x16x16xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x2x16x16xf32>

    // CHECK: [[MIN_WEIGHTS:%.*]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    // CHECK: [[MAX_WEIGHTS:%.*]] = const.Declare tensor<f32> = dense<1.000000e+01> : tensor<f32>
    // CHECK: [[FILTERS:%.*]] = const.Declare tensor<8x2x3x3xf32> = dense<5.000000e+00> : tensor<8x2x3x3xf32>
    // CHECK: [[FQ1:%.*]] = IE.FakeQuantize([[FILTERS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]], [[MIN_WEIGHTS]], [[MAX_WEIGHTS]]) {auto_broadcast = "NUMPY", levels = 255 : i64} : tensor<8x2x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<8x2x3x3xf32>

    // CHECK: [[EXPAND_DILATED:%.*]] = IE.ExpandDilated([[FQ1]]) {dilations = [2, 2]} : tensor<8x2x3x3xf32> -> tensor<8x2x5x5xf32>

    // CHECK: [[CONV:%.*]] = IE.Convolution([[FQ0]], [[EXPAND_DILATED]]) {dilations = [1, 1], pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x2x16x16xf32>, tensor<8x2x5x5xf32> -> tensor<1x8x16x16xf32>
}

// CHECK-LABEL: @LegalizeDilatedGroupConvolution
func @LegalizeDilatedGroupConvolution(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %filter = const.Declare tensor<3x1x3x3xf16> = dense<1.0> : tensor<3x1x3x3xf16>
    %0 = IE.GroupConvolution(%arg0, %filter)
        {
            dilations = [2, 2],
            groups = 3,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } :
        tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>
    return %0 : tensor<1x3x30x30xf16>

    // CHECK: [[FILTERS:%.*]] = const.Declare tensor<3x1x3x3xf16>
    // CHECK: [[EXPAND_DILATED:%.*]] = IE.ExpandDilated([[FILTERS]]) {dilations = [2, 2]} : tensor<3x1x3x3xf16> -> tensor<3x1x5x5xf16>
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution(%arg0, [[EXPAND_DILATED]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<3x1x5x5xf16> -> tensor<1x3x30x30xf16>

    // CHECK: return [[CONV]]
}
