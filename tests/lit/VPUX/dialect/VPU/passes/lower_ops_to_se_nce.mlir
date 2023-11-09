//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearHalfPixel([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearHalfPixel(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate
    
    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearFloatScales([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearFloatScales(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.999999e+00, 2.999999e+00],
            sizes_attr = [8, 8],
            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    return %0 : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate
    
    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearLargeScales([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x36x36xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearLargeScales(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x36x36xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [12.000000e+00, 12.0000000e+00],
            sizes_attr = [36, 36],
            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x36x36xf16, {order = #NHWC}>

    return %0 : tensor<1x16x36x36xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate
    
    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}
