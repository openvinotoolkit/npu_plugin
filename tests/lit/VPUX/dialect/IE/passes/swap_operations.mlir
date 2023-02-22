//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-operations  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithBias
func @SwapWithBias(%arg0: tensor<4x9728x1x1xf16>) -> tensor<1x512x4x1xf16> {
   %filter = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>
   %bias = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf16>

    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 512, 1]} : tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x512x1xf16> -> tensor<1x512x4x1xf16>
    %3 = IE.Add(%2, %bias) {auto_broadcast = "NUMPY"} : tensor<1x512x4x1xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x4x1xf16>

    return %3 : tensor<1x512x4x1xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
   // CHECK: IE.Transpose
   // CHECK-SAME: tensor<1x4x512x1xf16> -> tensor<1x512x4x1xf16>

  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithBiasNHWC
func @SwapWithBiasNHWC(%arg0: tensor<4x9728x1x1xf16>) -> tensor<1x512x1x4xf16> {
   %filter = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>
   %bias = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf16>

    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 512, 1]} : tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x4x512x1xf16> -> tensor<1x512x1x4xf16>
    %3 = IE.Add(%2, %bias) {auto_broadcast = "NUMPY"} : tensor<1x512x1x4xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x4xf16>

    return %3 : tensor<1x512x1x4xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
   // CHECK: IE.Transpose
   // CHECK-SAME: tensor<1x4x512x1xf16> -> tensor<1x512x1x4xf16>
  }

// -----

// CHECK-LABEL: @NotChangeSwapBiasBroadcastWithReshape
func @NotChangeSwapBiasBroadcastWithReshape(%arg0: tensor<1x924x77x1xf16>) -> tensor<1x12x77x77xf16> {
   %bias = const.Declare tensor<1x1x77x77xf16> = dense<1.000000e+00> : tensor<1x1x77x77xf16>

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 12, 77, 77]} : tensor<1x924x77x1xf16> -> tensor<1x12x77x77xf16>
    %1 = IE.Add(%0, %bias) {auto_broadcast = "NUMPY"} : tensor<1x12x77x77xf16>, tensor<1x1x77x77xf16> -> tensor<1x12x77x77xf16>

    return %1 : tensor<1x12x77x77xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<1x924x77x1xf16> -> tensor<1x12x77x77xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x12x77x77xf16>, tensor<1x1x77x77xf16> -> tensor<1x12x77x77xf16>
  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotChangeSwapBiasWithReorder
func @NotChangeSwapBiasWithReorder(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %bias) {auto_broadcast = "NUMPY"} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

   // CHECK: IE.Reorder
   // CHECK-SAME: tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
  }

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithRelu
func @SwapWithRelu(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.ReLU(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.ReLU
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>

}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithSigmoid
func @SwapWithSigmoid(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Sigmoid(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Sigmoid
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>

}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithTanh
func @SwapWithTanh(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Tanh(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Tanh
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>

}

// -----

// CHECK-LABEL: @NoSwapReshapeWithEltwise
func @NoSwapReshapeWithEltwise(%arg0: tensor<4x9728x1x1xf16>, %arg1: tensor<4x9728x1x1xf16>) -> tensor<1x4x1x512xf16> {
   %filter_0 = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>
   %filter_1 = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>

   %0 = IE.Convolution(%arg0, %filter_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   %1 = IE.Convolution(%arg1, %filter_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   %2 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4, 1, 512]} : tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   %3 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4, 1, 512]} : tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   %4 = IE.Add(%2, %3) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4x1x512xf16>, tensor<1x4x1x512xf16> -> tensor<1x4x1x512xf16>

   return %4 : tensor<1x4x1x512xf16>

   // CHECK: IE.Convolution(%arg0, %cst)
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.Convolution(%arg1, %cst)
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x4x1x512xf16>, tensor<1x4x1x512xf16> -> tensor<1x4x1x512xf16>
}

// -----

// CHECK-LABEL: @NoSwapReshapeWithLess4DBias
func @NoSwapReshapeWithLess4DBias(%arg0: tensor<4x16x1xf16>) -> tensor<1x4x1x512xf16> {
   %filter_0 = const.Declare tensor<512x16x1xf16> = dense<1.000000e+00> : tensor<512x16x1xf16>
   %bias = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x1x512xf16>

   %0 = IE.Convolution(%arg0, %filter_0) {dilations = [1], pads_begin = [0], pads_end = [0], strides = [1]} : tensor<4x16x1xf16>, tensor<512x16x1xf16> -> tensor<4x512x1xf16>
   %1 = IE.Reshape(%0) {shape_value = [1, 4, 1, 512]} : tensor<4x512x1xf16> -> tensor<1x4x1x512xf16>
   %2 = IE.Add(%1, %bias) {auto_broadcast = "NUMPY"} : tensor<1x4x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x4x1x512xf16>

   return %2 : tensor<1x4x1x512xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<4x16x1xf16>, tensor<512x16x1xf16> -> tensor<4x512x1xf16>
   // CHECK: IE.Reshape
   // CHECK-SAME: tensor<4x512x1xf16> -> tensor<1x4x1x512xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x4x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x4x1x512xf16>
}

// -----

// CHECK-LABEL: @NoSwapReshapeWithMoreThan4DSigmoid
func @NoSwapReshapeWithMoreThan4DSigmoid(%arg0: tensor<1x3x9x16x1xf16> ) -> tensor<3x9x16x1xf16> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [3, 9, 16, 1]} : tensor<1x3x9x16x1xf16> -> tensor<3x9x16x1xf16>
   %1 = IE.Sigmoid(%0) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>

   return %1 : tensor<3x9x16x1xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<1x3x9x16x1xf16> -> tensor<3x9x16x1xf16>
   // CHECK: IE.Sigmoid
   // CHECK-SAME: tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
}

// -----

// CHECK-LABEL: @OptimizeSliceTanH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func @OptimizeSliceTanH(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x8x32x32xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x8x32x32xf16>
   %1 = IE.Tanh(%0) : tensor<1x8x32x32xf16> -> tensor<1x8x32x32xf16>
   return %1 : tensor<1x8x32x32xf16>

   // CHECK:        [[TANH:%.+]] = IE.Tanh([[INPUT]]) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[TANH]] [0, 0, 0, 0] [1, 8, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x8x32x32xf16>
}

// -----

// CHECK-LABEL: @DoNotOptimizeSliceTanH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func @DoNotOptimizeSliceTanH(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x4x32x32xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   %1 = IE.Tanh(%0) : tensor<1x4x32x32xf16> -> tensor<1x4x32x32xf16>
   return %1 : tensor<1x4x32x32xf16>

   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   // CHECK:        [[TANH:%.+]] = IE.Tanh([[SLICE]]) : tensor<1x4x32x32xf16> -> tensor<1x4x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @OptimizeSigmoidReorder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16, {order = #NHWC}>
func @OptimizeSigmoidReorder(%arg0: tensor<1x16x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
   %1 = IE.Sigmoid(%0) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %1 : tensor<1x16x32x32xf16>

   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[INPUT]]) : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[SIGMOID]]) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @DoNotOptimizeSigmoidReorder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16, {order = #NHCW}>
func @DoNotOptimizeSigmoidReorder(%arg0: tensor<1x16x32x32xf16, {order = #NHCW}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
   %1 = IE.Sigmoid(%0) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %1 : tensor<1x16x32x32xf16>

   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[REORDER]]) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
}
