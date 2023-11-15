//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX30XX" --mempermute-processing %s | FileCheck %s

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MemPermuteProcessingWithoutFusing
func.func @MemPermuteProcessingWithoutFusing(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x64x32xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x16x32x64xf16> -> tensor<1x16x64x32xf16>

    return %2 : tensor<1x16x64x32xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute([[CONV]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x64x32xf16>

    // CHECK:   return [[PERMUTE]] : tensor<1x16x64x32xf16>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteProcessingConvertPass
func.func @MemPermuteProcessingConvertPass(%arg0: tensor<1x2x3x4xf32>,
                            %arg1: tensor<1x2x3x4xf32, {order = #NHWC}>,
                            %arg2: tensor<1x2x3x4xf32>) ->
                        (tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>) {
    %0 = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>

    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>

    %2 = IE.Reorder(%arg2) {dstOrder = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    return %0, %1, %2 : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>

    // CHECK-NOT: IE.Transpose
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Reorder
    // CHECK:     %[[VAL0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>
    // CHECK:     %[[VAL1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    // CHECK:     %[[VAL2:.*]] = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    // CHECK:     return %[[VAL0]], %[[VAL1:.*]], %[[VAL2:.*]] : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>
}
