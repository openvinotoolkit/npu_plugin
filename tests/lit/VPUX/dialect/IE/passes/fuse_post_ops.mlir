//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-post-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func @Conv2dWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @MaxPoolWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = "CEIL"
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.MaxPool
    // CHECK-SAME:     kernel_size = [2, 2]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     rounding_type = "CEIL"
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @DepthWiseConv2dWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x1x2x2xf16> = dense<1.0> : tensor<16x1x1x2x2xf16>, [#const.Reshape<[16, 1, 2, 2]>]
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            groups = 16,
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0]
        } :
        tensor<1x16x4x4xf16>, tensor<16x1x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.GroupConvolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     groups = 16
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @Conv2dWithClampTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.Clamp
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 1.000000e+00:127>
!qElemType1 = type !quant.uniform<u8<0:254>:f16, 1.0>
!qElemType2 = type !quant.uniform<u8:f16, 0.15748031466614967:128>

func @QuantizedConv2dWithClampTest(%arg0: tensor<1x16x20x20x!qElemType0>) -> tensor<1x32x20x20x!qElemType2> {
    %filters = const.Declare tensor<32x16x1x1x!qElemType1> = dense<1.0> : tensor<32x16x1x1xf32>,
                    [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]

    %0 = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x16x20x20x!qElemType0>, tensor<32x16x1x1x!qElemType1> -> tensor<1x32x20x20x!qElemType2>

    %1 = IE.Clamp(%0)
        {
            max = 5.000000e+00 : f64,
            min = -5.000000e+00 : f64
        } :
        tensor<1x32x20x20x!qElemType2> -> tensor<1x32x20x20x!qElemType2>

    return %1 : tensor<1x32x20x20x!qElemType2>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {max = 5.000000e+00 : f64, min = -5.000000e+00 : f64}, name = "IE.Clamp"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.Clamp
}

// -----

func @AddWithReLUTest() -> tensor<1x16x4x4xf16> {
    %0 = const.Declare tensor<1x16x4x4xf16> = dense<6.0> : tensor<1x16x4x4xf16>
    %1 = const.Declare tensor<1x16x4x4xf16> = dense<-7.0> : tensor<1x16x4x4xf16>
    %sum = IE.Add(%0, %1) { auto_broadcast = "NUMPY" } : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %relu = IE.ReLU(%sum) : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    return %relu : tensor<1x16x4x4xf16>

    // CHECK:       %[[RIGHT:.*]] = const.Declare tensor<1x16x4x4xf16> = dense<-7.000000e+00> : tensor<1x16x4x4xf16>
    // CHECK:       %[[LEFT:.*]] = const.Declare tensor<1x16x4x4xf16> = dense<6.000000e+00> : tensor<1x16x4x4xf16>
    // CHECK:       %[[SUM:.*]] = IE.Add(%[[LEFT]], %[[RIGHT]])
    // CHECK-SAME:     auto_broadcast = "NUMPY"
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-NOT:   IE.ReLU
}

// -----

func @AddWithLeakyReluTest() -> tensor<1x16x4x4xf16> {
    %0 = const.Declare tensor<1x16x4x4xf16> = dense<6.0> : tensor<1x16x4x4xf16>
    %1 = const.Declare tensor<1x16x4x4xf16> = dense<-7.0> : tensor<1x16x4x4xf16>
    %sum = IE.Add(%0, %1) { auto_broadcast = "NUMPY" } : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %leakyRelu = IE.LeakyRelu(%sum) {
            negative_slope = 0.100000e+00
        } : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    return %leakyRelu : tensor<1x16x4x4xf16>

    // CHECK:       %[[RIGHT:.*]] = const.Declare tensor<1x16x4x4xf16> = dense<-7.000000e+00> : tensor<1x16x4x4xf16>
    // CHECK:       %[[LEFT:.*]] = const.Declare tensor<1x16x4x4xf16> = dense<6.000000e+00> : tensor<1x16x4x4xf16>
    // CHECK:       %[[SUM:.*]] = IE.Add(%[[LEFT]], %[[RIGHT]])
    // CHECK:   IE.LeakyRelu
}

// -----

func @ShouldNotFuseScaleShiftTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<3.0> : tensor<1x16x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x16x3x3xf16>, tensor<1x16x1x1xf32> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:   IE.Convolution
    // CHECK:   IE.ScaleShift
}

// -----

func @Conv2dWithLeakyReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.000000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.LeakyRelu
}

// -----

func @Conv2dWithLeakyRelu15Test(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.500000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.LeakyRelu
}

// -----

func @Deconv2dWithLeakyReluTest(%arg0: tensor<1x32x64x100xf16>) -> tensor<1x16x128x101xf16> {
    %filters = const.Declare tensor<32x16x3x2xf16> = dense<1.0> : tensor<32x16x3x2xf16>
    %0 = IE.Deconvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            output_padding = [1, 0],
            pads_begin = [1, 0],
            pads_end = [1, 0],
            strides = [2, 1]
        } :
        tensor<1x32x64x100xf16>, tensor<32x16x3x2xf16> -> tensor<1x16x128x101xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x128x101xf16> -> tensor<1x16x128x101xf16>

    return %1 : tensor<1x16x128x101xf16>

    // CHECK:       IE.Deconvolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     output_padding = [1, 0]
    // CHECK-SAME:     pads_begin = [1, 0]
    // CHECK-SAME:     pads_end = [1, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.500000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [2, 1]
    // CHECK-NOT:   IE.LeakyRelu
}

// -----

func @Deconv2dWithLeakyReluNotFuseTest(%arg0: tensor<1x32x64x100xf16>, %arg1: tensor<32x16x3x2xf16>) -> tensor<1x16x128x101xf16> {
    %0 = IE.Deconvolution(%arg0, %arg1)
        {
            dilations = [1, 1],
            output_padding = [1, 0],
            pads_begin = [1, 0],
            pads_end = [1, 0],
            strides = [2, 1]
        } :
        tensor<1x32x64x100xf16>, tensor<32x16x3x2xf16> -> tensor<1x16x128x101xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x128x101xf16> -> tensor<1x16x128x101xf16>

    return %1 : tensor<1x16x128x101xf16>

    // CHECK:       IE.Deconvolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     output_padding = [1, 0]
    // CHECK-SAME:     pads_begin = [1, 0]
    // CHECK-SAME:     pads_end = [1, 0]
    // CHECK-SAME:     strides = [2, 1]
    // CHECK:     IE.LeakyRelu
}
