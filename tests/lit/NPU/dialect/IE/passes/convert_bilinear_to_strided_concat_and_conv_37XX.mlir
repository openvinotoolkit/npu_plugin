//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-bilinear-to-strided-concat-and-conv --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL: @ConvertInterpolateWithChannelNeedAlign
func.func @ConvertInterpolateWithChannelNeedAlign(%arg0: tensor<1x3x160x160xf16>) -> tensor<1x3x320x320xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x3x160x160xf16> -> tensor<1x3x320x320xf16>

    return %0 : tensor<1x3x320x320xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK:           [[INPUTREORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x3x160x160xf16> -> tensor<1x3x160x160xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT0:%.+]] = IE.Concat([[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x3x160x160xf16, {order = #NHWC}>, tensor<1x3x160x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x3x160x160xf16, {order = #NHWC}>, tensor<1x3x160x160xf16, {order = #NHWC}> -> tensor<1x3x640x160xf16, {order = #NHWC}>

    // CHECK:           [[SLICE0:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 3, 1, 160] : tensor<1x3x640x160xf16, {order = #NHWC}> to tensor<1x3x1x160xf16, {order = #NHWC}>
    // CHECK:           [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 639, 0] [1, 3, 1, 160] : tensor<1x3x640x160xf16, {order = #NHWC}> to tensor<1x3x1x160xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT1:%.+]] = IE.Concat([[SLICE0]], [[CONCAT0]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 641, 0]]} : tensor<1x3x1x160xf16, {order = #NHWC}>, tensor<1x3x640x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x3x1x160xf16, {order = #NHWC}> -> tensor<1x3x642x160xf16, {order = #NHWC}>

    // CHECK:           [[CONV0:%.+]] = IE.Convolution([[CONCAT1]], {{[^:]+}}) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} :
    // CHECK-SAME:      tensor<1x3x642x160xf16, {order = #NHWC}>, tensor<3x3x4x1xf16, {order = #NHWC}> -> tensor<1x3x320x160xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE0:%.+]] = IE.MemPermute([[CONV0]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x3x320x160xf16, {order = #NHWC}> -> tensor<1x3x160x320xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT2:%.+]] = IE.Concat([[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x3x160x320xf16, {order = #NHWC}>, tensor<1x3x160x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x3x160x320xf16, {order = #NHWC}>, tensor<1x3x160x320xf16, {order = #NHWC}> -> tensor<1x3x640x320xf16, {order = #NHWC}>

    // CHECK:           [[SLICE2:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 3, 1, 320] : tensor<1x3x640x320xf16, {order = #NHWC}> to tensor<1x3x1x320xf16, {order = #NHWC}>
    // CHECK:           [[SLICE3:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 639, 0] [1, 3, 1, 320] : tensor<1x3x640x320xf16, {order = #NHWC}> to tensor<1x3x1x320xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT3:%.+]] = IE.Concat([[SLICE2]], [[CONCAT2]], [[SLICE3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 641, 0]]} : tensor<1x3x1x320xf16, {order = #NHWC}>, tensor<1x3x640x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x3x1x320xf16, {order = #NHWC}> -> tensor<1x3x642x320xf16, {order = #NHWC}>

    // CHECK:           [[CONV1:%.+]] = IE.Convolution([[CONCAT3]], {{[^:]+}}) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} :
    // CHECK-SAME:      tensor<1x3x642x320xf16, {order = #NHWC}>, tensor<3x3x4x1xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE1:%.+]] = IE.MemPermute([[CONV1]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x3x320x320xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>
    // CHECK:           [[OUTPUTREORDER:%.+]] = IE.Reorder([[MEMPERMUTE1]]) {dstOrder = #NCHW} : tensor<1x3x320x320xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16>

    // CHECK:           return [[OUTPUTREORDER]] : tensor<1x3x320x320xf16>

}
