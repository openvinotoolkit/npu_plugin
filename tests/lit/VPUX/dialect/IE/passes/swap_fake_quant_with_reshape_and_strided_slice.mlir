//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-fake-quant-with-reshape-and-strided-slice %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @SwapFakeQuantReshape
func.func @SwapFakeQuantReshape(
        %input: tensor<1x1x40xf16>,
        %weights: tensor<512x40x1x1xf16>)
            -> tensor<1x512x1x1xf16> {
    %cst_0 = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    %cst_1 = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    %1 = IE.SoftMax(%input) {axisInd = 2} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    %2 = IE.AffineReshape(%1) {shape_value = [1, 1, 1, 40], dim_mapping = [[0], [1, 2], [3]]} : tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    %3 = IE.FakeQuantize(%2, %cst_0, %cst_1, %cst_0, %cst_1) {
                     auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                     levels = 256 : i64
                 } : tensor<1x1x1x40xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x40xf16>
    %4 = IE.AffineReshape(%3) {shape_value = [1, 40, 1, 1], dim_mapping = [[0], [0], [0], [1, 2, 3]]} : tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    %5 = IE.Convolution(%4, %weights) {
                    strides = [1, 1],
                    pads_begin = [0, 0],
                    pads_end = [0, 0],
                    dilations = [1, 1]
                } : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>

    return %5 : tensor<1x512x1x1xf16>

    // CHECK-DAG:       %[[FQ_MAX:.*]] = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    // CHECK-DAG:       %[[FQ_MIN:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK:       %[[SOFTMAX:.*]] = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    // CHECK:       %[[AFFINERESHAPE1:.*]] = IE.AffineReshape(%[[SOFTMAX]])
    // CHECK-SAME:      tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    // CHECK:       %[[AFFINERESHAPE2:.*]] = IE.AffineReshape(%[[AFFINERESHAPE1]])
    // CHECK-SAME:      tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    // CHECK:       %[[FQ:.*]] = IE.FakeQuantize(%[[AFFINERESHAPE2]], %[[FQ_MIN]], %[[FQ_MAX]], %[[FQ_MIN]], %[[FQ_MAX]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      tensor<1x40x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16>
    // CHECK-SAME:         -> tensor<1x40x1x1xf16>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[FQ]], %arg1)
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>
}

// -----

// CHECK-LABEL: @SwapFakeQuantWithStridedSlice
func.func @SwapFakeQuantWithStridedSlice(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x6x320x320xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<0.996688663> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16>
    %2 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %3 = IE.StridedSlice(%2) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %4 = IE.StridedSlice(%1) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    %5 = IE.StridedSlice(%4) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    %6 = IE.Concat(%3, %5) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x6x320x320xf16>

    return %6 : tensor<1x6x320x320xf16>

    // CHECK:      [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK:      [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.996688663> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK:      [[STRIDEDSLICE0:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    // CHECK:      [[STRIDEDSLICE1:%.*]] = IE.StridedSlice([[STRIDEDSLICE0]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:      [[FQ0:%.*]] = IE.FakeQuantize([[STRIDEDSLICE1]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x320x320xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x320x320xf16>

    // CHECK:      [[STRIDEDSLICE2:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16> -> tensor<1x3x320x640xf16>
    // CHECK:      [[STRIDEDSLICE3:%.*]] = IE.StridedSlice([[STRIDEDSLICE2]]) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16> -> tensor<1x3x320x320xf16>
    // CHECK:      [[FQ1:%.*]] = IE.FakeQuantize([[STRIDEDSLICE3]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x320x320xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x320x320xf16>

    // CHECK:      [[CONCAT:%.*]] = IE.Concat([[FQ0]], [[FQ1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]} : tensor<1x3x320x320xf16>, tensor<1x3x320x320xf16> -> tensor<1x6x320x320xf16>

    // CHECK:      return [[CONCAT]] : tensor<1x6x320x320xf16>
}
