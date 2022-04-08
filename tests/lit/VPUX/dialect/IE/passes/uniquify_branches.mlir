//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --uniquify-branches %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func @MoveExpandBeforeMultipleSlices(%arg0: tensor<2x70x4x4xf16>, %arg1: tensor<16x80x1x1xf16>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>

    %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    %3 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<2x70x4x4xf16> -> tensor<2x80x4x4xf16>
    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[EXPAND]] [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    // CHECK:   [[CONV0:%.+]] = IE.Convolution([[SLICE1]], %arg1)
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[SLICE0]], %arg1)
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]])

    // CHECK:   return [[CONCAT]] : tensor<2x16x4x4xf16>
}

// -----

func @NoChangesExpandModifyesSliceAxis(%arg0: tensor<1x140x4x4xf16>, %arg1: tensor<16x80x1x1xf16>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>
    %1 = IE.Slice %arg0 [0, 70, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>

    %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    %3 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [0, 70, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>

    // CHECK:   [[EXPAND0:%.+]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    // CHECK:   [[EXPAND1:%.+]] = IE.Expand([[SLICE1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @NoChangesDifferentExpands(%arg0: tensor<2x70x4x4xf16>, %arg1: tensor<16x80x1x1xf16>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>

    %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    %3 = IE.Expand(%1) {pads_begin = [0, 5, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>

    // CHECK:   [[EXPAND0:%.+]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    // CHECK:   [[EXPAND1:%.+]] = IE.Expand([[SLICE1]]) {pads_begin = [0, 5, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MoveReorderBeforeMultipleSlices(%arg0: tensor<2x80x4x4xf16>, %arg1: tensor<16x80x1x1xf16, {order = #NHWC}>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<2x80x4x4xf16> -> tensor<2x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[EXPAND]] [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16, {order = #NHWC}> to tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16, {order = #NHWC}> to tensor<1x80x4x4xf16, {order = #NHWC}>

    // CHECK:   [[CONV0:%.+]] = IE.Convolution([[SLICE1]], %arg1)
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[SLICE0]], %arg1)
    // CHECK:   [[CONCAT:%.+]] = IE.Concat(%3, %4)

    // CHECK:   return [[CONCAT]] : tensor<2x16x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @NoChangesReorderModifyesSliceAxis(%arg0: tensor<1x160x4x4xf16>, %arg1: tensor<16x80x1x1xf16, {order = #NHWC}>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [0, 80, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16> to tensor<1x80x4x4xf16>

    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16> to tensor<1x80x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [0, 80, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16> to tensor<1x80x4x4xf16>

    // CHECK:   [[REORDER0:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[REORDER1:%.+]] = IE.Reorder([[SLICE1]]) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func @NoChangesDifferentReorders(%arg0: tensor<2x80x4x4xf16>, %arg1: tensor<16x80x1x1xf16, {order = #NHWC}>) -> (tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<1x80x4x4xf16, {order = #NWHC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1) {dstOrder = #NWHC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NWHC}>

    return %2, %3: tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<1x80x4x4xf16, {order = #NWHC}>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    // CHECK:   [[REORDER0:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[REORDER1:%.+]] = IE.Reorder([[SLICE1]]) {dstOrder = #NWHC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NWHC}>
}

// -----

func @MoveTransposeBeforeMultipleSlices(%arg0: tensor<2x2x64x76xf16>, %arg1: tensor<1x64x1x1xf16>) -> tensor<4x76x1x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>
    %2 = IE.Slice %arg0 [1, 0, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>
    %3 = IE.Slice %arg0 [1, 1, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>

    %4 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %5 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %6 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %7 = IE.Transpose(%3) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>

    %8 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %9 = IE.Convolution(%arg1, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %10 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %11 = IE.Convolution(%arg1, %10) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %12 = IE.AffineReshape(%6) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %13 = IE.Convolution(%arg1, %12) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %14 = IE.AffineReshape(%7) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %15 = IE.Convolution(%arg1, %14) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %16 = IE.Concat(%9, %11, %13, %15) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} : tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16> -> tensor<4x76x1x1xf16>

    return %16: tensor<4x76x1x1xf16>

    // CHECK:       [[TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<2x2x64x76xf16> -> tensor<2x2x76x64xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[TRANSPOSE]] [1, 1, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[TRANSPOSE]] [1, 0, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>
    // CHECK:       [[SLICE3:%.*]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV0:%.*]] = IE.Convolution(%arg1, [[RESHAPE0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV1:%.*]] = IE.Convolution(%arg1, [[RESHAPE1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[RESHAPE2:%.*]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV2:%.*]] = IE.Convolution(%arg1, [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[RESHAPE3:%.*]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV3:%.*]] = IE.Convolution(%arg1, [[RESHAPE3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[CONCAT:%.*]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK-SAME{LITERAL}:                  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} :
    // CHECK-SAME:                  tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16> -> tensor<4x76x1x1xf16>

    // CHECK:       return [[CONCAT]] : tensor<4x76x1x1xf16>
}

// -----

func @NoChangesTransposeModifyesSliceAxis(%arg0: tensor<1x2x128x76xf16>, %arg1: tensor<1x64x1x1xf16>) -> tensor<2x76x1x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    %4 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %5 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>

    %8 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %9 = IE.Convolution(%arg1, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %10 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %11 = IE.Convolution(%arg1, %10) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %16 = IE.Concat(%9, %11) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16> -> tensor<2x76x1x1xf16>

    return %16: tensor<2x76x1x1xf16>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[SLICE0]]) {order_value = #NCWH} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    // CHECK:       [[TRANSPOSE1:%.*]] = IE.Transpose([[SLICE1]]) {order_value = #NCWH} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @NoChangesDifferentTransposes(%arg0: tensor<1x2x128x76xf16>, %arg1: tensor<1x64x1x1xf16>) -> (tensor<1x76x1x1xf16>, tensor<1x76x1x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    %4 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %5 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x76x1x64xf16>

    %8 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %9 = IE.Convolution(%arg1, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    return %9, %5: tensor<1x76x1x1xf16>, tensor<1x76x1x64xf16>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[SLICE0]]) {order_value = #NCWH} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    // CHECK:       [[TRANSPOSE1:%.*]] = IE.Transpose([[SLICE1]]) {order_value = #NWCH} : tensor<1x1x64x76xf16> -> tensor<1x76x1x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// If source has exactly one slice consumer we do not propagate transpose/layer.
func @NoChangesOneSliceConsumer(%arg0: tensor<1x2x128x76xf16>, %arg1: tensor<1x2x128x76xf16>) -> (tensor<1x1x76x64xf16>, tensor<1x1x76x64xf16>, tensor<1x2x128x76xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg1 [0, 1, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    %2 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %3 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>

    %4 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x2x128x76xf16> -> tensor<1x2x128x76xf16>

    return %2, %3, %4: tensor<1x1x76x64xf16>, tensor<1x1x76x64xf16>, tensor<1x2x128x76xf16>

    // CHECK: [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76]
    // CHECK: [[SLICE1:%.*]] = IE.Slice %arg1 [0, 1, 0, 0] [1, 1, 64, 76]

    // CHECK: [[TRANSPOSE0:%.*]] = IE.Transpose([[SLICE0]]) {order_value = #NCWH}
    // CHECK: [[TRANSPOSE1:%.*]] = IE.Transpose([[SLICE1]]) {order_value = #NCWH}

    // CHECK: [[SOFTMAX0:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64}

    // CHECK: return [[TRANSPOSE0]], [[TRANSPOSE1]], [[SOFTMAX0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MovePermuteCastBeforeMultipleSlices(%arg0: tensor<3x80x4x4xf16>) -> tensor<3x4x80x4xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %2 = IE.Slice %arg0 [2, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>

    %3 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %4 = IE.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %5 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>

    %6 = IE.Concat(%3, %4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    return %6: tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[SLICE0:%.*]] = IE.Slice [[PERMUTECAST]] [2, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[PERMUTECAST]] [1, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[SLICE2]], [[SLICE1]], [[SLICE0]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: return [[CONCAT0]] : tensor<3x4x80x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @NoChangesPermuteCastModifySameAxis(%arg0: tensor<1x3x5x4xf16>) -> tensor<3x5x4x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    %2 = IE.Slice %arg0 [0, 2, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>

    %3 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    %4 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    %5 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>

    %6 = IE.Concat(%3, %4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16> -> tensor<3x5x4x1xf16>

    return %6: tensor<3x5x4x1xf16>

    // CHECK: [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    // CHECK: [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    // CHECK: [[SLICE2:%.*]] = IE.Slice %arg0 [0, 2, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>

    // CHECK: [[PERMUTECAST0:%.*]] = IE.PermuteCast([[SLICE0]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    // CHECK: [[PERMUTECAST1:%.*]] = IE.PermuteCast([[SLICE1]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    // CHECK: [[PERMUTECAST2:%.*]] = IE.PermuteCast([[SLICE2]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[PERMUTECAST0]], [[PERMUTECAST1]], [[PERMUTECAST2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16> -> tensor<3x5x4x1xf16>

    // CHECK: return [[CONCAT]] : tensor<3x5x4x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MovePermuteCastBeforeMultipleSlicesAndNonSlice(%arg0: tensor<3x80x4x4xf16>) -> (tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %2 = IE.Slice %arg0 [2, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>

    %3 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    %5 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %6 = IE.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %7 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>

    %8 = IE.Concat(%5, %6, %7) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    return %4, %8: tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST0:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[SLICE0:%.*]] = IE.Slice [[PERMUTECAST0]] [2, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[PERMUTECAST0]] [1, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[PERMUTECAST0]] [0, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST1:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[REORDER:%.*]] = IE.Reorder([[PERMUTECAST1]]) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE2]], [[SLICE1]], [[SLICE0]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: return [[REORDER]], [[CONCAT]] : tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func @MoveAffineReshapeType1BeforeMultipleSlices(%arg0: tensor<1x2x76x64xf16>) -> (tensor<76x64xf16, {order = #CN}>, tensor<76x64xf16, {order = #CN}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>

    %2 = IE.AffineReshape(%0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [64, 76] } : tensor<1x1x76x64xf16> -> tensor<64x76xf16>
    %3 = IE.AffineReshape(%1) { dim_mapping = [[0], [0], [0], [1]], shape_value = [64, 76] } : tensor<1x1x76x64xf16> -> tensor<64x76xf16>

    %4 = IE.PermuteCast(%2) {dst_order = #CN, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #CN}>
    %5 = IE.PermuteCast(%3) {dst_order = #CN, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #CN}>

    return %4, %5: tensor<76x64xf16, {order = #CN}>, tensor<76x64xf16, {order = #CN}>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 76]} : tensor<1x2x76x64xf16> -> tensor<128x76xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %0 [64, 0] [64, 76] : tensor<128x76xf16> to tensor<64x76xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice %0 [0, 0] [64, 76] : tensor<128x76xf16> to tensor<64x76xf16>
    // CHECK:       [[PERMUTE0:%.*]] = IE.PermuteCast([[SLICE0]]) {dst_order = #map, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #map}>
    // CHECK:       [[PERMUTE1:%.*]] = IE.PermuteCast([[SLICE1]]) {dst_order = #map, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #map}>
    // CHECK:       return [[PERMUTE0]], [[PERMUTE1]] : tensor<76x64xf16, {order = #map}>, tensor<76x64xf16, {order = #map}>

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MoveAffineReshapeType2BeforeMultipleSlices(%arg0: tensor<152x64xf16>) -> (tensor<1x1x64x76xf16, {order = #NHWC}>, tensor<1x1x64x76xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0] [76, 64] : tensor<152x64xf16> to tensor<76x64xf16>
    %1 = IE.Slice %arg0 [76, 0] [76, 64] : tensor<152x64xf16> to tensor<76x64xf16>

    %2 = IE.AffineReshape(%0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 64, 76] } : tensor<76x64xf16> -> tensor<1x1x64x76xf16>
    %3 = IE.AffineReshape(%1) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 64, 76] } : tensor<76x64xf16> -> tensor<1x1x64x76xf16>

    %4 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x1x64x76xf16> -> tensor<1x1x64x76xf16, {order = #NHWC}>
    %5 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x1x64x76xf16> -> tensor<1x1x64x76xf16, {order = #NHWC}>

    return %4, %5: tensor<1x1x64x76xf16, {order = #NHWC}>, tensor<1x1x64x76xf16, {order = #NHWC}>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [2, 1, 64, 76]} : tensor<152x64xf16> -> tensor<2x1x64x76xf16>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[RESHAPE]]) {dstOrder = #NHWC} : tensor<2x1x64x76xf16> -> tensor<2x1x64x76xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [1, 1, 64, 76] : tensor<2x1x64x76xf16, {order = #NHWC}> to tensor<1x1x64x76xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[REORDER]] [1, 0, 0, 0] [1, 1, 64, 76] : tensor<2x1x64x76xf16, {order = #NHWC}> to tensor<1x1x64x76xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE0]], [[SLICE1]] : tensor<1x1x64x76xf16, {order = #NHWC}>, tensor<1x1x64x76xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MoveAffineReshapeType3BeforeMultipleSlices(%arg0: tensor<1x2x76x64xf16>) -> (tensor<76x64x1x1xf16, {order = #NHWC}>, tensor<76x64x1x1xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>

    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %4 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<76x64x1x1xf16> -> tensor<76x64x1x1xf16, {order = #NHWC}>
    %5 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<76x64x1x1xf16> -> tensor<76x64x1x1xf16, {order = #NHWC}>
    return %4, %5: tensor<76x64x1x1xf16, {order = #NHWC}>, tensor<76x64x1x1xf16, {order = #NHWC}>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [152, 64, 1, 1]} : tensor<1x2x76x64xf16> -> tensor<152x64x1x1xf16>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[RESHAPE]]) {dstOrder = #NHWC} : tensor<152x64x1x1xf16> -> tensor<152x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [76, 64, 1, 1] : tensor<152x64x1x1xf16, {order = #NHWC}> to tensor<76x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[REORDER]] [76, 0, 0, 0] [76, 64, 1, 1] : tensor<152x64x1x1xf16, {order = #NHWC}> to tensor<76x64x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE0]], [[SLICE1]] : tensor<76x64x1x1xf16, {order = #NHWC}>, tensor<76x64x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MoveAffineReshapeBeforeMultipleSlicesForNHWCLayout(%arg0: tensor<1x2x76x64xf16, {order = #NHWC}>) -> (tensor<2x2x19x64xf16>, tensor<2x2x19x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16, {order = #NHWC}> to tensor<1x2x38x64xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [0, 0, 38, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16, {order = #NHWC}> to tensor<1x2x38x64xf16, {order = #NHWC}>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [0, 2], [3]], shape_value = [2, 2, 19, 64]} : tensor<1x2x38x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16, {order = #NHWC}>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [0, 2], [3]], shape_value = [2, 2, 19, 64]} : tensor<1x2x38x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16, {order = #NHWC}>
    %4 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<2x2x19x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16>
    %5 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<2x2x19x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16>
    return %4, %5: tensor<2x2x19x64xf16>, tensor<2x2x19x64xf16>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [0, 2], [3]], shape_value = [4, 2, 19, 64]} : tensor<1x2x76x64xf16, {order = #NHWC}> -> tensor<4x2x19x64xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[RESHAPE]]) {dstOrder = #NCHW} : tensor<4x2x19x64xf16, {order = #NHWC}> -> tensor<4x2x19x64xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [2, 2, 19, 64] : tensor<4x2x19x64xf16> to tensor<2x2x19x64xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[REORDER]] [2, 0, 0, 0] [2, 2, 19, 64] : tensor<4x2x19x64xf16> to tensor<2x2x19x64xf16>
    // CHECK:       return [[SLICE0]], [[SLICE1]] : tensor<2x2x19x64xf16>, tensor<2x2x19x64xf16>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func @NoChangesAffineReshapeSliceOnOtherDimension(%arg0: tensor<1x2x76x64xf16>) -> (tensor<64x76xf16, {order = #CN}>, tensor<64x76xf16, {order = #CN}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>
    %1 = IE.Slice %arg0 [0, 0, 38, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>

    %2 = IE.AffineReshape(%0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64] } : tensor<1x2x38x64xf16> -> tensor<76x64xf16>
    %3 = IE.AffineReshape(%1) { dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64] } : tensor<1x2x38x64xf16> -> tensor<76x64xf16>

    %4 = IE.PermuteCast(%2) {dst_order = #CN, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #CN}>
    %5 = IE.PermuteCast(%3) {dst_order = #CN, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #CN}>

    return %4, %5: tensor<64x76xf16, {order = #CN}>, tensor<64x76xf16, {order = #CN}>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 0, 38, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>
    // CHECK:       [[RESHAPE0:%.*]] = IE.AffineReshape(%0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64]} : tensor<1x2x38x64xf16> -> tensor<76x64xf16>
    // CHECK:       [[RESHAPE1:%.*]] = IE.AffineReshape(%1)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64]} : tensor<1x2x38x64xf16> -> tensor<76x64xf16>
    // CHECK:       [[PERMUTE0:%.*]] = IE.PermuteCast(%2) {dst_order = #map, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #map}>
    // CHECK:       [[PERMUTE1:%.*]] = IE.PermuteCast(%3) {dst_order = #map, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #map}>
    // CHECK:       return [[PERMUTE0]], [[PERMUTE1]] : tensor<64x76xf16, {order = #map}>, tensor<64x76xf16, {order = #map}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MoveMemPermuteBeforeMultipleSlicesAndNonSlice(%arg0: tensor<3x80x4x4xf16>) -> (tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %2 = IE.Slice %arg0 [2, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>

    %3 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    %5 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %6 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %7 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>

    %8 = IE.Concat(%5, %6, %7) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    return %4, %8: tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE0:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[SLICE1:%.*]] = IE.Slice [[MEMPERMUTE0]] [2, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[MEMPERMUTE0]] [1, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE3:%.*]] = IE.Slice [[MEMPERMUTE0]] [0, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE1:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[REORDER:%.*]] = IE.Reorder([[MEMPERMUTE1]]) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE3]], [[SLICE2]], [[SLICE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    // CHECK: return [[REORDER]], [[CONCAT]] : tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>
}
