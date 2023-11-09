//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-strided-slice-to-dwconv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertStridedSlice2DWConv
func.func @ConvertStridedSlice2DWConv(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}> {
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x640xf16, {order = #NHWC}>

    // CHECK-NOT: IE.StridedSlice
    // CHECK: [[CST0:%.*]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>]
    // CHECK: [[GRP_CONV:%.*]] = IE.GroupConvolution(%arg0, [[CST0]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x1x1x1xf16> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK: return [[GRP_CONV]] : tensor<1x3x320x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotConvertForUnalignedChannels
func.func @DoNotConvertForUnalignedChannels(%arg0: tensor<1x3x320x640xf16, {order = #NHWC}>) -> tensor<1x3x320x320xf16, {order = #NHWC}> {
    %1 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x320xf16, {order = #NHWC}>

    // CHECK: [[STRIDED_SLICE:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 320, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x320x640xf16, {order = #NHWC}> -> tensor<1x3x320x320xf16, {order = #NHWC}>
    // CHECK: [[STRIDED_SLICE]] : tensor<1x3x320x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertStridedSliceWithFQ2DWConv
func.func @ConvertStridedSliceWithFQ2DWConv(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<3.18300e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.068850e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16, {order = #NHWC}>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x640xf16, {order = #NHWC}>

    // CHECK-NOT: IE.StridedSlice
    // CHECK: [[CST:%.*]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>]
    // CHECK: [[CST0:%.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK: [[CST1:%.*]] = const.Declare tensor<f16> = dense<2.540000e+02> : tensor<f16>
    // CHECK: [[CST2:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.183590e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[CST3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<3.068360e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize(%arg0, [[CST3]], [[CST2]], [[CST3]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16, {order = #NHWC}>
    // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize([[CST]], [[CST0]], [[CST1]], [[CST0]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<3x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<3x1x1x1xf16>
    // CHECK: [[GRP_CONV:%.*]] = IE.GroupConvolution([[FQ_0]], [[FQ_1]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x1x1x1xf16> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    // CHECK: return [[GRP_CONV]] : tensor<1x3x320x640xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotConvertStridedSliceIfOutputNCHWToResult
func.func @NotConvertStridedSliceIfOutputNCHWToResult(%arg0: tensor<1x3x640x640xf16, {order = #NCHW}>) -> tensor<1x3x320x640xf16> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NCHW}> -> tensor<1x3x320x640xf16>
    return %0 : tensor<1x3x320x640xf16>

    // CHECK: [[SLICE:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NCHW}> -> tensor<1x3x320x640xf16>
    // CHECK: return [[SLICE]] : tensor<1x3x320x640xf16
}
