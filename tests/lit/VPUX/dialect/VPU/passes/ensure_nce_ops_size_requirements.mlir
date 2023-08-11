//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x16x16x!qElemType0, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x16x16x!qElemType0, {order = #NHWC}>) -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<9216x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<9216x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<9216x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [9216, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<4608x1x1x4xsi32> = dense<10>
    // CHECK-SAME:      tensor<9216x1x1x4xsi32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 4]>]

    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<9216x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[4608, 0, 0, 0], [4608, 32, 3, 3]>]

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<4608x1x1x4xsi32> = dense<10>
    // CHECK-SAME:      : tensor<9216x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 4]>]

    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      tensor<9216x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [4608, 32, 3, 3]>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [4608, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [4608, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 4608, 0, 0]
    // CHECK-SAME:          -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverIH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x8704x16x!qElemType0, {order = #NHWC}>
func.func @SplitQuantNCEConvOverIH(%arg0: tensor<1x32x8704x16x!qElemType0, {order = #NHWC}>) -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [64, 32, 3, 3],
        strides = [2, 2]
    } -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<64x1x1x4xsi32>

    // CHECK:        [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 4352, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType0, {order = #NHWC}> to tensor<1x32x4352x16x!qElemType0, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 0 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x64x2176x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4351, 0] [1, 32, 4353, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType0, {order = #NHWC}> to tensor<1x32x4353x16x!qElemType0, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x64x2176x8x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 2176, 0]
    // CHECK-SAME:          -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC2Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC2Convs(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x9728x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x1536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 8192, 0, 0], [512, 1536, 1, 1]>]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 8192, 1, 1]>]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8192, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x8192x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE0:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [512, 8192, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 8192, 0, 0] [1, 1536, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x1536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE1:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [512, 1536, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}
