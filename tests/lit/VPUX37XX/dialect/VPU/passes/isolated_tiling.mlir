//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --isolated-tiling --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func @SplitSwConvOverOC(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>)
            -> tensor<1x256x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE0:%.+]] = VPU.Slice [[BIAS]] [0, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE1:%.+]] = VPU.Slice [[BIAS]] [0, 128, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE1]], [[BIAS_TILE1]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----
// CHECK-LABEL: func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16>
func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x200x200xf16>)
            -> tensor<1x16x200x200xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x200x200xf16> -> tensor<1x16x200x200xf16>
    return %1 : tensor<1x16x200x200xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16> to tensor<1x16x101x200xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x100x200xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16> to tensor<1x16x101x200xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x100x200xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16>
}

// -----

// CHECK-LABEL: func @SplitSwAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x2048x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func @SplitSwAddOverC(
        %input1: tensor<1x2048x14x14xf16>,
        %input2: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = "NUMPY" } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Add([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Add([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func @SplitAddSameInputOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func @SplitAddSameInputOverC(
        %input: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.And(%input, %input) { auto_broadcast = "NUMPY" } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:       : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.And([[INPUT_TILE0]], [[INPUT_TILE0]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.And([[INPUT_TILE1]], [[INPUT_TILE1]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

// -----

func @InterpSplitOverH(
        %input1: tensor<1x32x64x64xf16>)
            -> tensor<1x32x256x256xf16> {

    %0 = const.Declare tensor<2xsi64> = dense<[256, 256]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[4.000000e+00, 4.00000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>

    %3 = VPU.Interpolate(%input1, %0, %1, %2) {
            attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01, mode = "LINEAR", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
            operand_segment_sizes = dense<1> : vector<4xi32> } :
        tensor<1x32x64x64xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x32x256x256xf16>

    return %3 : tensor<1x32x256x256xf16>
}

// CHECK-LABEL: func @InterpSplitOverH
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x32x64x64xf16>

// Tile 0

// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x86x256xf16>

// Tile 1

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 21, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x85x256xf16>

// Tile 2

// CHECK:       [[TILE2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 42, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x85x256xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]
// CHECK-SAME:      -> tensor<1x32x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x32x256x256xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
func @NoTilingClusterNCEConv(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN}>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights_table as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1]
            } -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:        [[CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:          %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHTS]] as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHT_TABLE]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           [[NCE_CONV:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:              pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[NCE_CONV]]

    // CHECK:         return [[CLUSTER_TILING]] : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

// CHECK-LABEL: func @GatherSplit
func @GatherSplit(%arg0: tensor<4004x320xf16>) -> tensor<1x320xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4004x320xf16>, tensor<1xsi32> -> tensor<1x320xf16>
  return %0 : tensor<1x320xf16>

  // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.ConvertElemType<si32>]

  // Tile 0
  // CHECK:     [[Tile0:%.+]] = VPU.Slice %arg0 [0, 0] [4004, 160] : tensor<4004x320xf16> to tensor<4004x160xf16>
  // CHECK:     [[Gather0:%.+]] = VPU.Gather([[Tile0]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2]} : tensor<4004x160xf16>, tensor<1xsi32> -> tensor<1x160xf16>
  
  // Tile 1
  // CHECK:     [[Tile1:%.+]] = VPU.Slice %arg0 [0, 160] [4004, 160] : tensor<4004x320xf16> to tensor<4004x160xf16>
  // CHECK:     [[Gather1:%.+]] = VPU.Gather([[Tile1]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2]} : tensor<4004x160xf16>, tensor<1xsi32> -> tensor<1x160xf16>
  
  // CHECK:     [[Concat:%.+]] = VPU.Concat([[Gather0]], [[Gather1]]) 
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0], [0, 160]]} : tensor<1x160xf16>, tensor<1x160xf16> -> tensor<1x320xf16>

  // CHECK:     return [[Concat]] 
}

// -----

// CHECK-LABEL: func @GatherSplitWithBatchDims
func @GatherSplitWithBatchDims(%arg0: tensor<2x4004x320xf16>) -> tensor<2x1x320xf16> {
  %cst = const.Declare tensor<2x1xsi32> = dense<[[-4004], [4003]]> : tensor<2x1xsi64>, [#const.ConvertElemType<si32>]
  %0 = VPU.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<2x4004x320xf16>, tensor<2x1xsi32> -> tensor<2x1x320xf16>
  return %0 : tensor<2x1x320xf16>

  // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<2x1xsi32>

  // Tile 0
  // CHECK:     [[Tile0:%.+]] = VPU.Slice %arg0 [0, 0, 0] [2, 4004, 107] : tensor<2x4004x320xf16> to tensor<2x4004x107xf16>
  // CHECK:     [[Gather0:%.+]] = VPU.Gather([[Tile0]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 1 : i64, tilingStrategy = [1, 1, 3]} : tensor<2x4004x107xf16>, tensor<2x1xsi32> -> tensor<2x1x107xf16>

  // Tile 1
  // CHECK:     [[Tile1:%.+]] = VPU.Slice %arg0 [0, 0, 107] [2, 4004, 107] : tensor<2x4004x320xf16> to tensor<2x4004x107xf16>
  // CHECK:     [[Gather1:%.+]] = VPU.Gather([[Tile1]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 1 : i64, tilingStrategy = [1, 1, 3]} : tensor<2x4004x107xf16>, tensor<2x1xsi32> -> tensor<2x1x107xf16>

  // Tile 2
  // CHECK:     [[Tile2:%.+]] = VPU.Slice %arg0 [0, 0, 214] [2, 4004, 106] : tensor<2x4004x320xf16> to tensor<2x4004x106xf16>
  // CHECK:     [[Gather2:%.+]] = VPU.Gather([[Tile2]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 1 : i64, tilingStrategy = [1, 1, 3]} : tensor<2x4004x106xf16>, tensor<2x1xsi32> -> tensor<2x1x106xf16>

  // CHECK:    [[Concat:%.+]] = VPU.Concat([[Gather0]], [[Gather1]], [[Gather2]]) 
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 107], [0, 0, 214]]} : tensor<2x1x107xf16>, tensor<2x1x107xf16>, tensor<2x1x106xf16> -> tensor<2x1x320xf16>

  // CHECK:     return [[Concat]] 
}

// -----
// CHECK-LABEL: func @Yuv2RGBSplit
func @Yuv2RGBSplit(%arg0: tensor<1x993x982x1xf16>) -> tensor<1x662x982x3xf16> {
  %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x662x982x1xf16>
  %1 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
  %2 = VPU.Reshape(%1) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>
  %3 = VPU.YuvToRgb(%0, %2) {inFmt = "NV12", operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = "RGB"} : tensor<1x662x982x1xf16>, tensor<1x331x491x2xf16> -> tensor<1x662x982x3xf16>
  return %3 : tensor<1x662x982x3xf16>

    // CHECK:    %0 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
    // CHECK:    %1 = VPU.Reshape(%0) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>

    // Tile 0
    // CHECK:    %2 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 220, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x220x982x1xf16>
    // CHECK:    %3 = VPU.Slice %1 [0, 0, 0, 0] [1, 110, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x110x491x2xf16>
    // CHECK:    %4 = VPU.YuvToRgb(%2, %3) {inFmt = "NV12", operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = "RGB", tilingStrategy = [1, 3, 1, 1]} : tensor<1x220x982x1xf16>, tensor<1x110x491x2xf16> -> tensor<1x220x982x3xf16>

    // Tile 1
    // CHECK:    %5 = VPU.Slice %arg0 [0, 220, 0, 0] [1, 220, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x220x982x1xf16>
    // CHECK:    %6 = VPU.Slice %1 [0, 110, 0, 0] [1, 110, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x110x491x2xf16>
    // CHECK:    %7 = VPU.YuvToRgb(%5, %6) {inFmt = "NV12", operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = "RGB", tilingStrategy = [1, 3, 1, 1]} : tensor<1x220x982x1xf16>, tensor<1x110x491x2xf16> -> tensor<1x220x982x3xf16>

    // Tile 2
    // CHECK:    %8 = VPU.Slice %arg0 [0, 440, 0, 0] [1, 220, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x220x982x1xf16>
    // CHECK:    %9 = VPU.Slice %1 [0, 220, 0, 0] [1, 110, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x110x491x2xf16>
    // CHECK:    %10 = VPU.YuvToRgb(%8, %9) {inFmt = "NV12", operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = "RGB", tilingStrategy = [1, 3, 1, 1]} : tensor<1x220x982x1xf16>, tensor<1x110x491x2xf16> -> tensor<1x220x982x3xf16>

    // Tile 3
    // CHECK:    %11 = VPU.Slice %arg0 [0, 660, 0, 0] [1, 2, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x2x982x1xf16>
    // CHECK:    %12 = VPU.Slice %1 [0, 330, 0, 0] [1, 1, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x1x491x2xf16>
    // CHECK:    %13 = VPU.YuvToRgb(%11, %12) {inFmt = "NV12", operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = "RGB", tilingStrategy = [1, 3, 1, 1]} : tensor<1x2x982x1xf16>, tensor<1x1x491x2xf16> -> tensor<1x2x982x3xf16>

    // CHECK:    %14 = VPU.Concat(%4, %7, %10, %13)
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 220, 0, 0], [0, 440, 0, 0], [0, 660, 0, 0]]} : tensor<1x220x982x3xf16>, tensor<1x220x982x3xf16>, tensor<1x220x982x3xf16>, tensor<1x2x982x3xf16> -> tensor<1x662x982x3xf16>
    // CHECK:    return %14 : tensor<1x662x982x3xf16>
}

// -----

// CHECK-LABEL: func @GatherNDSplit
func @GatherNDSplit(%arg0: tensor<3x5x512x512xf16>) -> tensor<3x1x100x512xf16> {
    %cst = const.Declare tensor<3x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<3x5x512x512xf16>, tensor<3x1x100x2xsi32> -> tensor<3x1x100x512xf16>
    return %0 : tensor<3x1x100x512xf16>

    // CHECK: [[Indices_2:%.+]] = const.Declare tensor<1x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>, [#const.SubView<[2, 0, 0, 0], [1, 1, 100, 2]>]
    // CHECK: [[Indices_1:%.+]] = const.Declare tensor<1x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>, [#const.SubView<[1, 0, 0, 0], [1, 1, 100, 2]>]
    // CHECK: [[Indices_0:%.+]] = const.Declare tensor<1x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 100, 2]>]

    // CHECK: [[Tile0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND0:%.+]] = VPU.GatherND([[Tile0]], [[Indices_0]]) {batch_dims = 1 : i64, tilingStrategy = [3, 1, 1, 2]} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 256] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND1:%.+]] = VPU.GatherND([[Tile1]], [[Indices_0]]) {batch_dims = 1 : i64, tilingStrategy = [3, 1, 1, 2]} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile2:%.+]] = VPU.Slice %arg0 [1, 0, 0, 0] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND2:%.+]] = VPU.GatherND([[Tile2]], [[Indices_1]]) {batch_dims = 1 : i64, tilingStrategy = [3, 1, 1, 2]} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile3:%.+]] = VPU.Slice %arg0 [1, 0, 0, 256] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND3:%.+]] = VPU.GatherND([[Tile3]], [[Indices_1]]) {batch_dims = 1 : i64, tilingStrategy = [3, 1, 1, 2]} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile4:%.+]] = VPU.Slice %arg0 [2, 0, 0, 0] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND4:%.+]] = VPU.GatherND([[Tile4]], [[Indices_2]]) {batch_dims = 1 : i64, tilingStrategy = [3, 1, 1, 2]} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile5:%.+]] = VPU.Slice %arg0 [2, 0, 0, 256] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND5:%.+]] = VPU.GatherND([[Tile5]], [[Indices_2]]) {batch_dims = 1 : i64, tilingStrategy = [3, 1, 1, 2]} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[GatherND0]], [[GatherND1]], [[GatherND2]], [[GatherND3]], [[GatherND4]], [[GatherND5]])
    // CHECK-SAME: [0, 0, 0, 0], [0, 0, 0, 256], [1, 0, 0, 0], [1, 0, 0, 256], [2, 0, 0, 0], [2, 0, 0, 256]
    // CHECK-SAME: -> tensor<3x1x100x512xf16>

    // CHECK: return [[OUTPUT]] : tensor<3x1x100x512xf16>
}

// -----

// CHECK-LABEL: func @GatherNDSplitIndices
func @GatherNDSplitIndices(%arg0: tensor<64x2xf16>) -> tensor<400000x2xf16> {
    %cst = const.Declare tensor<400000x1xsi32> = dense<1> : tensor<400000x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<400000x1xsi32> -> tensor<400000x2xf16>
    return %0 : tensor<400000x2xf16>

    // CHECK: [[Indices_1:%.+]] = const.Declare tensor<200000x1xsi32> = dense<1> : tensor<400000x1xsi32>, [#const.SubView<[200000, 0], [200000, 1]>]
    // CHECK: [[Indices_0:%.+]] = const.Declare tensor<200000x1xsi32> = dense<1> : tensor<400000x1xsi32>, [#const.SubView<[0, 0], [200000, 1]>]

    // CHECK: [[GatherND0:%.+]] = VPU.GatherND(%arg0, [[Indices_0]]) {batch_dims = 0 : i64, tilingStrategy = [2, 1]} : tensor<64x2xf16>, tensor<200000x1xsi32> -> tensor<200000x2xf16>
    // CHECK: [[GatherND1:%.+]] = VPU.GatherND(%arg0, [[Indices_1]]) {batch_dims = 0 : i64, tilingStrategy = [2, 1]} : tensor<64x2xf16>, tensor<200000x1xsi32> -> tensor<200000x2xf16>

    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[GatherND0]], [[GatherND1]])
    // CHECK-SAME: [0, 0], [200000, 0]
    // CHECK-SAME: -> tensor<400000x2xf16>

    // CHECK: return [[OUTPUT]] : tensor<400000x2xf16>
}

// -----

// CHECK-LABEL: func @DepthToSpaceBlocksFirstSplit
func @DepthToSpaceBlocksFirstSplit(%arg0: tensor<1x480x10x120xf32>) -> tensor<1x30x40x480xf32> {
  %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32> -> tensor<1x480x10x120xf16>
  %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = "BLOCKS_FIRST"} : tensor<1x480x10x120xf16> -> tensor<1x30x40x480xf16>
  %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16> -> tensor<1x30x40x480xf32>
  return %2 : tensor<1x30x40x480xf32>

  // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32> to tensor<1x240x10x120xf32>
  // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x240x10x120xf32> -> tensor<1x240x10x120xf16>
  
  // CHECK:     %2 = VPU.Slice %arg0 [0, 240, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32> to tensor<1x240x10x120xf32>
  // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x240x10x120xf32> -> tensor<1x240x10x120xf16>
  
  // CHECK:     %4 = VPU.Concat(%1, %3) {
  // CHECK-SAME:[0, 0, 0, 0], [0, 240, 0, 0]
  // CHECK-SAME:-> tensor<1x480x10x120xf16>
  
  // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16> to tensor<1x480x5x120xf16>
  // CHECK:     %6 = VPU.DepthToSpace(%5) {block_size = 4 : i64, mode = "BLOCKS_FIRST", tilingStrategy = [1, 1, 2, 1]} : tensor<1x480x5x120xf16> -> tensor<1x30x20x480xf16>
  
  // CHECK:     %7 = VPU.Slice %4 [0, 0, 5, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16> to tensor<1x480x5x120xf16>
  // CHECK:     %8 = VPU.DepthToSpace(%7) {block_size = 4 : i64, mode = "BLOCKS_FIRST", tilingStrategy = [1, 1, 2, 1]} : tensor<1x480x5x120xf16> -> tensor<1x30x20x480xf16>
  
  // CHECK:     %9 = VPU.Concat(%6, %8)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf16>

  // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 30, 20, 480] : tensor<1x30x40x480xf16> to tensor<1x30x20x480xf16>
  // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32, tilingStrategy = [1, 1, 2, 1]} : tensor<1x30x20x480xf16> -> tensor<1x30x20x480xf32>
  
  // CHECK:     %12 = VPU.Slice %9 [0, 0, 20, 0] [1, 30, 20, 480] : tensor<1x30x40x480xf16> to tensor<1x30x20x480xf16>
  // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32, tilingStrategy = [1, 1, 2, 1]} : tensor<1x30x20x480xf16> -> tensor<1x30x20x480xf32>
  
  // CHECK:     %14 = VPU.Concat(%11, %13) 
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf32>
  
  // CHECK:     return %14 : tensor<1x30x40x480xf32>
}

// -----

// CHECK-LABEL: func @DepthToSpaceDepthFirstSplit
func @DepthToSpaceDepthFirstSplit(%arg0: tensor<1x480x10x120xf32>) -> tensor<1x30x40x480xf32> {
  %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32> -> tensor<1x480x10x120xf16>
  %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x480x10x120xf16> -> tensor<1x30x40x480xf16>
  %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16> -> tensor<1x30x40x480xf32>
  return %2 : tensor<1x30x40x480xf32>

  // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32> to tensor<1x240x10x120xf32>
  // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x240x10x120xf32> -> tensor<1x240x10x120xf16>
  
  // CHECK:     %2 = VPU.Slice %arg0 [0, 240, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32> to tensor<1x240x10x120xf32>
  // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x240x10x120xf32> -> tensor<1x240x10x120xf16>
  
  // CHECK:     %4 = VPU.Concat(%1, %3) {
  // CHECK-SAME:[0, 0, 0, 0], [0, 240, 0, 0]
  // CHECK-SAME:-> tensor<1x480x10x120xf16>
  
  // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf16> to tensor<1x240x10x120xf16>
  // CHECK:     %6 = VPU.DepthToSpace(%5) {block_size = 4 : i64, mode = "DEPTH_FIRST", tilingStrategy = [1, 2, 1, 1]} : tensor<1x240x10x120xf16> -> tensor<1x15x40x480xf16>

  // CHECK:     %7 = VPU.Slice %4 [0, 240, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf16> to tensor<1x240x10x120xf16>
  // CHECK:     %8 = VPU.DepthToSpace(%7) {block_size = 4 : i64, mode = "DEPTH_FIRST", tilingStrategy = [1, 2, 1, 1]} : tensor<1x240x10x120xf16> -> tensor<1x15x40x480xf16>

  // CHECK:     %9 = VPU.Concat(%6, %8)
  // CHECK-SAME:[0, 0, 0, 0], [0, 15, 0, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf16>

  // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 30, 20, 480] : tensor<1x30x40x480xf16> to tensor<1x30x20x480xf16>
  // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32, tilingStrategy = [1, 1, 2, 1]} : tensor<1x30x20x480xf16> -> tensor<1x30x20x480xf32>
  
  // CHECK:     %12 = VPU.Slice %9 [0, 0, 20, 0] [1, 30, 20, 480] : tensor<1x30x40x480xf16> to tensor<1x30x20x480xf16>
  // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32, tilingStrategy = [1, 1, 2, 1]} : tensor<1x30x20x480xf16> -> tensor<1x30x20x480xf32>
  
  // CHECK:     %14 = VPU.Concat(%11, %13) 
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf32>
  
  // CHECK:     return %14 : tensor<1x30x40x480xf32>
}

// -----

// CHECK-LABEL:   func @SpaceToDepthBlockFirstSplit
func @SpaceToDepthBlockFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
      %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
      %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = "BLOCKS_FIRST"} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
      %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
      return %2 : tensor<1x768x40x20xf32>

      // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
      // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

      // CHECK:     %2 = VPU.Slice %arg0 [0, 0, 80, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
      // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

      // CHECK:     %4 = VPU.Concat(%1, %3)
      // CHECK-SAME:[0, 0, 0, 0], [0, 0, 80, 0]
      // CHECK-SAME:-> tensor<1x48x160x80xf16>

      // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 48, 160, 40]  : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
      // CHECK:     %6 = VPU.SpaceToDepthOp(%5) {block_size = 4 : i64, mode = "BLOCKS_FIRST", tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

      // CHECK:     %7 = VPU.Slice %4 [0, 0, 0, 40]  [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
      // CHECK:     %8 = VPU.SpaceToDepthOp(%7) {block_size = 4 : i64, mode = "BLOCKS_FIRST", tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

      // CHECK:     %9 = VPU.Concat(%6, %8)
      // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 10]
      // CHECK-SAME:-> tensor<1x768x40x20xf16>

      // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
      // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

      // CHECK:     %12 = VPU.Slice %9 [0, 384, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
      // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

      // CHECK:     %14 = VPU.Concat(%11, %13)
      // CHECK-SAME:[0, 0, 0, 0], [0, 384, 0, 0]
      // CHECK-SAME:-> tensor<1x768x40x20xf32>

      // CHECK:     return %14 : tensor<1x768x40x20xf32>
}

// -----

// CHECK-LABEL: func @SpaceToDepthDepthFirstSplit
func @SpaceToDepthDepthFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
    // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

    // CHECK:     %2 = VPU.Slice %arg0 [0, 0, 80, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
    // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

    // CHECK:     %4 = VPU.Concat(%1, %3)
    // CHECK-SAME:[0, 0, 0, 0], [0, 0, 80, 0]
    // CHECK-SAME:-> tensor<1x48x160x80xf16>

    // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 48, 160, 40]  : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:     %6 = VPU.SpaceToDepthOp(%5) {block_size = 4 : i64, mode = "DEPTH_FIRST", tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

    // CHECK:     %7 = VPU.Slice %4 [0, 0, 0, 40]  [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:     %8 = VPU.SpaceToDepthOp(%7) {block_size = 4 : i64, mode = "DEPTH_FIRST", tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

    // CHECK:     %9 = VPU.Concat(%6, %8)
    // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 10]
    // CHECK-SAME:-> tensor<1x768x40x20xf16>

    // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
    // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

    // CHECK:     %12 = VPU.Slice %9 [0, 384, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
    // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

    // CHECK:     %14 = VPU.Concat(%11, %13)
    // CHECK-SAME:[0, 0, 0, 0], [0, 384, 0, 0]
    // CHECK-SAME:-> tensor<1x768x40x20xf32>

    // CHECK:     return %14 : tensor<1x768x40x20xf32>
  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = type !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>
func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>) -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<10>
    // CHECK-SAME:      : tensor<512x1x1x4xsi32>, [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]

    // CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<256x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [256, 32, 3, 3]>]

    // CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<10>
    // CHECK-SAME:      : tensor<512x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]

    // CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<256x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [256, 32, 3, 3]>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:          -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}>
    // CHECK-SAME:      = dense<10> : tensor<16x1x1x4xsi32>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>
func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x1024x24x24xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x24x24xf16, {order = #NHWC}>)
            -> tensor<1x1024x24x24xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x24xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 512, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 512, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:      -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x24x24xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16, {order = #NHWC}>
func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x2048x14x14xf16, {order = #NHWC}>) -> tensor<1x2048x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x14x14xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE0]], [[INPUT_TILE0]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE1]], [[INPUT_TILE1]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ConvertU8F32SplitOverH(%arg0: tensor<1x2x80x4000xui8, {order = #NHWC}>) -> tensor<1x2x80x4000xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x4000xui8, {order = #NHWC}> -> tensor<1x2x80x4000xf32, {order = #NHWC}>
  return %0 : tensor<1x2x80x4000xf32, {order = #NHWC}>
}

// CHECK-LABEL: @ConvertU8F32SplitOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x4000xui8, {order = #NHWC}>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 40, 4000]
// CHECK-SAME:      : tensor<1x2x80x4000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x40x4000xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convert([[INPUT_TILE0]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x40x4000xf32, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 2, 40, 4000]
// CHECK-SAME:      : tensor<1x2x80x4000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x40x4000xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convert([[INPUT_TILE1]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x40x4000xf32, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:      -> tensor<1x2x80x4000xf32, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x2x80x4000xf32, {order = #NHWC}>

// -----

func @SigmoidSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SigmoidSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sigmoid([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sigmoid([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @TanhSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Tanh(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @TanhSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tanh([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tanh([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @ExpSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Exp(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ExpSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Exp([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Exp([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SqrtSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SqrtSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sqrt([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sqrt([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @EluSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @EluSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Elu([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1], x = 1.000000e+00 : f64} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Elu([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1], x = 1.000000e+00 : f64} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @ClampSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ClampSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]]) {
// CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]]) {
// CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @ReLUSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ReLUSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @HSwishSplitOverH(%arg0: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {
  %0 = VPU.HSwish(%arg0) : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>
  return %0 : tensor<1x16x80x1280xf16>
}

// CHECK-LABEL: @HSwishSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSwish([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 20, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSwish([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.HSwish([[INPUT_TILE2]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 60, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.HSwish([[INPUT_TILE3]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0], [0, 0, 60, 0]
// CHECK-SAME:  : tensor<1x16x20x1280xf16>, tensor<1x16x20x1280xf16>, tensor<1x16x20x1280xf16>,
// CHECK-SAME:   tensor<1x16x20x1280xf16> -> tensor<1x16x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x80x1280xf16>

// -----

func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Divide([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Divide([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {
  %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>
  return %0 : tensor<1x40x40x546xf16>
}
// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 546, 20, 40]
// CHECK-SAME:  : tensor<1x546x40x40xf16> to tensor<1x546x20x40xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MemPermute([[INPUT_TILE0]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 2, 1, 1]
// CHECK-SAME:  } : tensor<1x546x20x40xf16> -> tensor<1x20x40x546xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 20, 0] [1, 546, 20, 40]
// CHECK-SAME:  : tensor<1x546x40x40xf16> to tensor<1x546x20x40xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MemPermute([[INPUT_TILE1]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 2, 1, 1]
// CHECK-SAME:  } : tensor<1x546x20x40xf16> -> tensor<1x20x40x546xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 20, 0, 0]
// CHECK-SAME:  : tensor<1x20x40x546xf16>, tensor<1x20x40x546xf16> -> tensor<1x40x40x546xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x40x40x546xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func @AvgPoolSwSplit2Part(%arg0: tensor<1x32x1800x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
  %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x32x1800x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  return %0 : tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
}
// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x1800x16xf16, {order = #NHWC}>) -> tensor<1x32x1789x16xf16, {order = #NHWC}> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 906, 16]
// CHECK-SAME:  :  tensor<1x32x1800x16xf16, {order = #NHWC}> to tensor<1x32x906x16xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
// CHECK-SAME:  } : tensor<1x32x906x16xf16, {order = #NHWC}> -> tensor<1x32x895x16xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 895, 0] [1, 32, 905, 16]
// CHECK-SAME:  : tensor<1x32x1800x16xf16, {order = #NHWC}> to tensor<1x32x905x16xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
// CHECK-SAME:  } : tensor<1x32x905x16xf16, {order = #NHWC}> -> tensor<1x32x894x16xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 895, 0]
// CHECK-SAME:  : tensor<1x32x895x16xf16, {order = #NHWC}>, tensor<1x32x894x16xf16, {order = #NHWC}> -> tensor<1x32x1789x16xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x32x1789x16xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80xf16, {order = #NHWC}>
func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x160x80x80xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x160x80x80xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[WEIGHTS_TILE:%.+]] =  const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_SM_TILE]], [[WEIGHTS_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80xf16, {order = #NHWC}> to tensor<1x32x41x80xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x80xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80xf16, {order = #NHWC}> to tensor<1x32x41x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_2]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x80xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = type !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitSparseQuantNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80x!qElemType0, {order = #NHWC}>
func @SplitSparseQuantNCEConvOverOH(%arg0: tensor<1x32x80x80x!qElemType0, {order = #NHWC}>) -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [320, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<320x1x1x4xsi32>

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<320x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80x!qElemType0, {order = #NHWC}> to tensor<1x32x41x80x!qElemType0, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_0]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x320x40x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80x!qElemType0, {order = #NHWC}> to tensor<1x32x41x80x!qElemType0, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x320x40x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x23040xf16, {order = #NHWC}>
func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x23040xf16, {order = #NHWC}>) -> tensor<1x16x1x23040xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [2.500000e-01]}, strides = [1, 1]} -> tensor<1x16x1x23040xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x23040xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 7680]
    // CHECK-SAME:      tensor<1x16x7x23040xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 7680] [1, 16, 7, 7680]
    // CHECK-SAME:      tensor<1x16x7x23040xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 15360] [1, 16, 7, 7680]
    // CHECK-SAME:      tensor<1x16x7x23040xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 7680], [0, 0, 0, 15360]
    // CHECK-SAME:      -> tensor<1x16x1x23040xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x23040xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16, {order = #NHWC}>
func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16, {order = #NHWC}>) -> tensor<1x1x1x368640xf16, {order = #NHWC}> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x1x7x368640xf16, {order = #NHWC}> -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x368640xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 122880] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 245760] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.AvgPool([[INPUT_TILE2]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 122880], [0, 0, 0, 245760]
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func @GridSampleSplit
func @GridSampleSplit(%arg0: tensor<1x3x272x480xf16>, %arg1: tensor<1x272x480x2xf16>) -> tensor<1x3x272x480xf16> {
      %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = "BILINEAR", padding_mode = "BORDER"} : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>
      return %0 : tensor<1x3x272x480xf16>

      // Tile 0

      // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 2, 272, 480] : tensor<1x3x272x480xf16> to tensor<1x2x272x480xf16>

      // CHECK:     %1 = VPU.GridSample(%0, %arg1) {align_corners, mode = "BILINEAR", padding_mode = "BORDER", tilingStrategy = [1, 2, 1, 1]} : tensor<1x2x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x2x272x480xf16>

      // Tile 1

      // CHECK:     %2 = VPU.Slice %arg0 [0, 2, 0, 0] [1, 1, 272, 480] : tensor<1x3x272x480xf16> to tensor<1x1x272x480xf16>

      // CHECK:     %3 = VPU.GridSample(%2, %arg1) {align_corners, mode = "BILINEAR", padding_mode = "BORDER", tilingStrategy = [1, 2, 1, 1]} : tensor<1x1x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x1x272x480xf16>

      // CHECK:     %4 = VPU.Concat(%1, %3)
      // CHECK-SAME:[0, 0, 0, 0], [0, 2, 0, 0]
      // CHECK-SAME:-> tensor<1x3x272x480xf16>

      // CHECK:     return %4 : tensor<1x3x272x480xf16>

}
