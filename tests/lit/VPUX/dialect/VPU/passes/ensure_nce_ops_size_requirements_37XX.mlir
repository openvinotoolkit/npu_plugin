//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEAveragePoolOverOW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x3x8832xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverOW(%arg0: tensor<1x16x3x8832xf16, {order = #NHWC}>) -> tensor<1x16x1x8832xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [3, 1],
        multiClusterStrategy = "Clustering", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64,
               clamp_low = -2147483648 : i64,
               fp_prelu_alpha = 1.000000e+00 : f64,
               lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "NOOP",
               quant_scale = [0.33333333333333331]},
               strides = [1, 1]}
        -> tensor<1x16x1x8832xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_0]]) {kernel_size = [3, 1], multiClusterStrategy = "Clustering",
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:              mode = "NOOP", quant_scale = [0.33333333333333331]},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4416] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_1]]) {kernel_size = [3, 1], multiClusterStrategy = "Clustering",
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:              mode = "NOOP", quant_scale = [0.33333333333333331]},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 4416]
    // CHECK-SAME:          -> tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8832xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC3Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3Convs(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 16384, 0, 0], [512, 256, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 8192, 0, 0], [512, 8192, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 8192, 1, 1]>]
  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8192, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x8192x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE0:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [512, 8192, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 8192, 0, 0] [1, 8192, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x8192x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [512, 8192, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 16384, 0, 0] [1, 256, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x256x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE2:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [512, 256, 1, 1], strides = [1, 1]} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise(%1, %3) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD", quant_scale = [1.000000e+00]}} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0:%.+]], [[CONV_OUT2:%.+]]) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}} -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverICandOC
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverICandOC(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x9216x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<9216x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    rawFilterShape = [9216, 16640, 1, 1],
    strides = [1, 1]
  } -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<4608x1x1x4xsi32> =
  // CHECK-SAME:  : tensor<9216x1x1x4xsi32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 4]>]
  // CHECK:       [[FILTER0:%.+]] = const.Declare tensor<4608x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 16384, 0, 0], [9216, 256, 1, 1]>, #const.SubView<[4608, 0, 0, 0], [4608, 256, 1, 1]>]
  // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<4608x1x1x4xsi32> =
  // CHECK-SAME:  : tensor<9216x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 4]>]
  // CHECK:       [[FILTER1:%.+]] = const.Declare tensor<4608x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 16384, 0, 0], [9216, 256, 1, 1]>, #const.SubView<[0, 0, 0, 0], [4608, 256, 1, 1]>]
  // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<4608x1x1x4xsi32> =
  // CHECK-SAME:  : tensor<9216x1x1x4xsi32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 4]>]
  // CHECK:       [[FILTER2:%.+]] = const.Declare tensor<4608x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 8192, 0, 0], [9216, 8192, 1, 1]>, #const.SubView<[4608, 0, 0, 0], [4608, 8192, 1, 1]>]
  // CHECK:       [[WEIGHTS_TABLE3:%.+]] = const.Declare tensor<4608x1x1x4xsi32> =
  // CHECK-SAME:  : tensor<9216x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 4]>]
  // CHECK:       [[FILTER3:%.+]] = const.Declare tensor<4608x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 8192, 0, 0], [9216, 8192, 1, 1]>, #const.SubView<[0, 0, 0, 0], [4608, 8192, 1, 1]>]
  // CHECK:       [[WEIGHTS_TABLE4:%.+]] = const.Declare tensor<4608x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [9216, 8192, 1, 1]>, #const.SubView<[4608, 0, 0, 0], [4608, 8192, 1, 1]>]
  // CHECK:       [[FILTER4:%.+]] = const.Declare tensor<4608x8192x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [9216, 8192, 1, 1]>, #const.SubView<[0, 0, 0, 0], [4608, 8192, 1, 1]>]
  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 8192, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x8192x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER4:%.+]], [[WEIGHTS_TABLE3:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [4608, 8192, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[WEIGHTS_TABLE4:%.+]], [[WEIGHTS_TABLE2:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [4608, 8192, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT0:%.+]] = VPU.Concat([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice %arg0 [0, 8192, 0, 0] [1, 8192, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x8192x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER3:%.+]], [[WEIGHTS_TABLE3:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [4608, 8192, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT3:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [4608, 8192, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT1:%.+]] = VPU.Concat([[CONV_OUT2:%.+]], [[CONV_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice %arg0 [0, 16384, 0, 0] [1, 256, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x256x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT4:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [4608, 256, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT5:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [4608, 256, 1, 1], strides = [1, 1]} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[CONV_OUT4:%.+]], [[CONV_OUT5:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE3:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE4:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise(%12, %13) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD", quant_scale = [1.000000e+00]}} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE5:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE6:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise(%15, %16) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD", quant_scale = [1.000000e+00]}} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[ADD_OUT0:%.+]], [[ADD_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE7:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE8:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT2:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE7:%.+]], [[INPUT_SLICE8:%.+]]) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE9:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE10:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT3:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE9:%.+]], [[INPUT_SLICE10:%.+]]) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}} -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT3:%.+]] = VPU.Concat([[ADD_OUT2:%.+]], [[ADD_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       return [[CONCAT_OUT3:%.+]] : tensor<1x9216x4x1xf16, {order = #NHWC}>
}
