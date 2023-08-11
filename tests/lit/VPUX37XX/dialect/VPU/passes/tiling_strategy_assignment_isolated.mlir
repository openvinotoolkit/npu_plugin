//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s

// CHECK-LABEL: func.func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func.func @SplitSwConvOverOC(
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

    // CHECK:       [[CONV:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[CONV]] : tensor<1x256x64x64xf16>
}

// -----
// CHECK-LABEL: func.func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16>
func.func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x200x200xf16>)
            -> tensor<1x16x200x200xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x200x200xf16> -> tensor<1x16x200x200xf16>
    return %1 : tensor<1x16x200x200xf16>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16>

    // CHECK:       return [[MAXPOOL]] : tensor<1x16x200x200xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x2048x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitSwAddOverC(
        %input1: tensor<1x2048x14x14xf16>,
        %input2: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // CHECK:       [[ADD:%.+]] = VPU.Add([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[ADD]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func.func @SplitAddSameInputOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitAddSameInputOverC(
        %input: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.And(%input, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // CHECK:       [[AND:%.+]] = VPU.And([[INPUT]], [[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[AND]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x32x64x64xf16>)
            -> tensor<1x32x256x256xf16> {

    %0 = const.Declare tensor<2xsi64> = dense<[256, 256]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[4.000000e+00, 4.00000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>

    %3 = VPU.Interpolate(%input1, %0, %1, %2) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            operand_segment_sizes = dense<1> : vector<4xi32> } :
        tensor<1x32x64x64xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x32x256x256xf16>

    return %3 : tensor<1x32x256x256xf16>

    // CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]]
    // CHECK-SAME:      tilingStrategy = [1, 3, 1, 1]
    // CHECK-SAME:      : tensor<1x32x64x64xf16>
    // CHECK-SAME:      -> tensor<1x32x256x256xf16>

    // CHECK:       return [[INTERP0]] : tensor<1x32x256x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
func.func @InterpSplitOverH(
    %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
            -> tensor<1x64x192x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operand_segment_sizes = dense<[1, 0, 0, 0]> :
        vector<4xi32>,
        sizes_attr = [192, 320],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
    return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 6, 1]
    // CHECH-SAME:  : tensor<1x64x48x80xf16, {order = #NHWC}>
    // CHECH-SAME:  -> tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x192x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x35x35xf16, {order = #NHWC}>
func.func @InterpSplitOverHW(
    %input1: tensor<1x128x35x35xf16, {order = #NHWC}>)
            -> tensor<1x128x168x335xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%input1) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operand_segment_sizes = dense<[1, 0, 0, 0]> :
        vector<4xi32>,
        sizes_attr = [168, 335]} :
        tensor<1x128x35x35xf16, {order = #NHWC}> -> tensor<1x128x168x335xf16, {order = #NHWC}>
    return %0 : tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 7, 5]
    // CHECH-SAME:  : tensor<1x128x35x35xf16, {order = #NHWC}>
    // CHECH-SAME:  -> tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x128x168x335xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCNoCommonFactor
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x31x31xf16, {order = #NHWC}>
func.func @InterpSplitOverCNoCommonFactor(
    %arg0: tensor<1x64x31x31xf16, {order = #NHWC}>)
            -> tensor<1x64x121x121xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operand_segment_sizes = dense<[1, 0, 0, 0]> :
        vector<4xi32>,
        sizes_attr = [121, 121],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x31x31xf16, {order = #NHWC}> -> tensor<1x64x121x121xf16, {order = #NHWC}>
    return %0 : tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
    // CHECH-SAME:  : tensor<1x64x31x31xf16, {order = #NHWC}>
    // CHECH-SAME:  -> tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x121x121xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
func.func @NoTilingClusterNCEConv(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
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

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-DAG:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:        [[CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:          %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHTS]] as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHT_TABLE]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           [[NCE_CONV:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:              pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-NOT:               tilingStrategy
    // CHECK-SAME:              -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[NCE_CONV]]

    // CHECK:         return [[CLUSTER_TILING]] : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @GatherSplit
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4004x320xf16>
func.func @GatherSplit(%arg0: tensor<4004x320xf16>) -> tensor<1x320xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4004x320xf16>, tensor<1xsi32> -> tensor<1x320xf16>
    return %0 : tensor<1x320xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.ConvertElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2]} : tensor<4004x320xf16>, tensor<1xsi32> -> tensor<1x320xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @GatherSplitWithBatchDims
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x4004x320xf16>
func.func @GatherSplitWithBatchDims(%arg0: tensor<2x4004x320xf16>) -> tensor<2x1x320xf16> {
    %cst = const.Declare tensor<2x1xsi32> = dense<1> : tensor<2x1xsi64>, [#const.ConvertElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<2x4004x320xf16>, tensor<2x1xsi32> -> tensor<2x1x320xf16>
    return %0 : tensor<2x1x320xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<2x1xsi32> = dense<1> : tensor<2x1xsi64>, [#const.ConvertElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 1 : i64, tilingStrategy = [2, 1, 2]} : tensor<2x4004x320xf16>, tensor<2x1xsi32> -> tensor<2x1x320xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @GatherSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<387072x3xf16>
func.func @GatherSplitOptimize(%arg0: tensor<387072x3xf16>) -> tensor<1x387072x3xf16> {
    %cst = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.ConvertElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>
    return %0 : tensor<1x387072x3xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.ConvertElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2, 3]} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @Yuv2RGBSplit
func.func @Yuv2RGBSplit(%arg0: tensor<1x993x982x1xf16>) -> tensor<1x662x982x3xf16> {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x662x982x1xf16>
    %1 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>
    %3 = VPU.YuvToRgb(%0, %2) {inFmt = #IE.color_fmt<NV12>, operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = #IE.color_fmt<RGB>} : tensor<1x662x982x1xf16>, tensor<1x331x491x2xf16> -> tensor<1x662x982x3xf16>
    return %3 : tensor<1x662x982x3xf16>

    // CHECK:    [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x662x982x1xf16>
    // CHECK:    [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
    // CHECK:    [[RESHAPE:%.+]] = VPU.Reshape([[SLICE1]]) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>
    // CHECK:    [[YUV2RGB:%.+]] = VPU.YuvToRgb([[SLICE0]], [[RESHAPE]]) {inFmt = #IE.color_fmt<NV12>, operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, outFmt = #IE.color_fmt<RGB>, tilingStrategy = [1, 3, 1, 1]} : tensor<1x662x982x1xf16>, tensor<1x331x491x2xf16> -> tensor<1x662x982x3xf16>
    // CHECK:    return [[YUV2RGB]] : tensor<1x662x982x3xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplit
func.func @GatherNDSplit(%arg0: tensor<3x5x512x512xf16>) -> tensor<3x1x100x512xf16> {
    %cst = const.Declare tensor<3x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<3x5x512x512xf16>, tensor<3x1x100x2xsi32> -> tensor<3x1x100x512xf16>
    return %0 : tensor<3x1x100x512xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<3x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherND(%arg0, [[CST]]) {
    // CHECK-SAME:               batch_dims = 1 : i64,
    // CHECK-SAME:               tilingStrategy = [3, 1, 1, 2]}
    // CHECK-SAME:           : tensor<3x5x512x512xf16>, tensor<3x1x100x2xsi32> -> tensor<3x1x100x512xf16>

    // CHECK: return [[GATHER]] : tensor<3x1x100x512xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitIndices
func.func @GatherNDSplitIndices(%arg0: tensor<64x2xf16>) -> tensor<400000x2xf16> {
    %cst = const.Declare tensor<400000x1xsi32> = dense<1> : tensor<400000x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<400000x1xsi32> -> tensor<400000x2xf16>
    return %0 : tensor<400000x2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<400000x1xsi32> = dense<1> : tensor<400000x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherND(%arg0, [[CST]]) {
    // CHECK-SAME:               batch_dims = 0 : i64,
    // CHECK-SAME:               tilingStrategy = [2, 1]}
    // CHECK-SAME:           : tensor<64x2xf16>, tensor<400000x1xsi32> -> tensor<400000x2xf16>

    // CHECK: return [[GATHER]] : tensor<400000x2xf16>
}

// -----

// CHECK-LABEL: func.func @DepthToSpaceBlocksFirstSplit
func.func @DepthToSpaceBlocksFirstSplit(%arg0: tensor<1x480x10x120xf32>) -> tensor<1x30x40x480xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32> -> tensor<1x480x10x120xf16>
    %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x10x120xf16> -> tensor<1x30x40x480xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16> -> tensor<1x30x40x480xf32>
    return %2 : tensor<1x30x40x480xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x480x10x120xf32> -> tensor<1x480x10x120xf16>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x480x10x120xf16> -> tensor<1x30x40x480xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[D2S]]) {dstElemType = f32, tilingStrategy = [1, 1, 1, 2]} : tensor<1x30x40x480xf16> -> tensor<1x30x40x480xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x30x40x480xf32>
}

// -----

// CHECK-LABEL: func.func @DepthToSpaceDepthFirstSplit
func.func @DepthToSpaceDepthFirstSplit(%arg0: tensor<1x480x10x120xf32>) -> tensor<1x30x40x480xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32> -> tensor<1x480x10x120xf16>
    %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x10x120xf16> -> tensor<1x30x40x480xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16> -> tensor<1x30x40x480xf32>
    return %2 : tensor<1x30x40x480xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x480x10x120xf32> -> tensor<1x480x10x120xf16>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, tilingStrategy = [1, 2, 1, 1]} : tensor<1x480x10x120xf16> -> tensor<1x30x40x480xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[D2S]]) {dstElemType = f32, tilingStrategy = [1, 1, 1, 2]} : tensor<1x30x40x480xf16> -> tensor<1x30x40x480xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x30x40x480xf32>
}

// -----

// CHECK-LABEL:   func.func @SpaceToDepthBlockFirstSplit
func.func @SpaceToDepthBlockFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       [[S2D:%.+]] = VPU.SpaceToDepthOp([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[S2D]]) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x768x40x20xf32>
}

// -----

// CHECK-LABEL: func.func @SpaceToDepthDepthFirstSplit
func.func @SpaceToDepthDepthFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       [[S2D:%.+]] = VPU.SpaceToDepthOp([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[S2D]]) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x768x40x20xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        return [[CONV]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>) -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<512x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [512, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:          -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[CONV]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}>
    // CHECK-SAME:      = dense<10> : tensor<16x1x1x4xsi32>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:      tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddOverC(
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

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    // CHECK:       return [[ELTWISE]] : tensor<1x1024x24x24xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x2048x14x14xf16, {order = #NHWC}>) -> tensor<1x2048x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[ELTWISE]] : tensor<1x2048x14x14xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertU8F32SplitOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x4000xui8, {order = #NHWC}>
func.func @ConvertU8F32SplitOverW(%arg0: tensor<1x2x80x4000xui8, {order = #NHWC}>) -> tensor<1x2x80x4000xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x4000xui8, {order = #NHWC}> -> tensor<1x2x80x4000xf32, {order = #NHWC}>
    return %0 : tensor<1x2x80x4000xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[INPUT]]) {
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      }> -> tensor<1x2x80x4000xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT]] : tensor<1x2x80x4000xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SigmoidSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sigmoid([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME       : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @TanhSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @TanhSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Tanh(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Tanh([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}


// -----

// CHECK-LABEL: @ExpSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @ExpSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
    // CHECK:       [[OUTPUT:%.+]] = VPU.Exp([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SqrtSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SqrtSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sqrt([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @EluSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @EluSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Elu([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2], x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Clamp([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @HSwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16>
func.func @HSwishSplitOverW(%arg0: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {
    %0 = VPU.HSwish(%arg0) : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>
    return %0 : tensor<1x16x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.HSwish([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 4]} : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Divide([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16>
func.func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {
    %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>
    return %0 : tensor<1x40x40x546xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MemPermute([[INPUT]]) {
    // CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:  } : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x40x40x546xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x1800x16xf16, {order = #NHWC}>) -> tensor<1x32x1789x16xf16, {order = #NHWC}>
func.func @AvgPoolSwSplit2Part(%arg0: tensor<1x32x1800x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x1800x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    return %0 : tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]]) {
    // CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:  } : tensor<1x32x1800x16xf16, {order = #NHWC}> -> tensor<1x32x1789x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x1789x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80xf16, {order = #NHWC}>
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x160x80x80xf16, {order = #NHWC}> {
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

    // CHECK:        [[WEIGHTS:%.+]] =  const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x80xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitSparseQuantNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80x!qElemType0, {order = #NHWC}>
func.func @SplitSparseQuantNCEConvOverOH(%arg0: tensor<1x32x80x80x!qElemType0, {order = #NHWC}>) -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}> {
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

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<320x1x1x384xi1>, is_weights

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<320x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x23040xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x23040xf16, {order = #NHWC}>) -> tensor<1x16x1x23040xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [2.500000e-01]}, strides = [1, 1]} -> tensor<1x16x1x23040xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x23040xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x23040xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x23040xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16, {order = #NHWC}>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16, {order = #NHWC}>) -> tensor<1x1x1x368640xf16, {order = #NHWC}> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x368640xf16, {order = #NHWC}> -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x368640xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceForward
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x1x1xf16>
func.func @SplitGRUSequenceForward(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x1x1xf16>) -> (tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>) {
      %cst = const.Declare tensor<1x3x10xf16> = dense<1.000000e+00> : tensor<1x3x10xf16>
      %cst_0 = const.Declare tensor<1x3x1xf16> = dense<1.000000e+00> : tensor<1x3x1xf16>
      %cst_1 = const.Declare tensor<1x4xf16> = dense<1.000000e+00> : tensor<1x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x10xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x10xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x4xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x4xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]])
    // CHECK-SAME:          tilingStrategy = [2, 1, 2, 1]
    // CHECK-SAME:     : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceReverse
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x1x1xf16>
func.func @SplitGRUSequenceReverse(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x1x1xf16>) -> (tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>) {
      %cst = const.Declare tensor<1x3x10xf16> = dense<1.000000e+00> : tensor<1x3x10xf16>
      %cst_0 = const.Declare tensor<1x3x1xf16> = dense<1.000000e+00> : tensor<1x3x1xf16>
      %cst_1 = const.Declare tensor<1x4xf16> = dense<1.000000e+00> : tensor<1x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x10xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x10xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x4xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x4xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]])
    // CHECK-SAME:          tilingStrategy = [2, 1, 2, 1]
    // CHECK-SAME:     : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceBidirectional
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x2x1xf16>
func.func @SplitGRUSequenceBidirectional(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x2x1xf16>) -> (tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>) {
      %cst = const.Declare tensor<2x3x10xf16> = dense<1.000000e+00> : tensor<2x3x10xf16>
      %cst_0 = const.Declare tensor<2x3x1xf16> = dense<1.000000e+00> : tensor<2x3x1xf16>
      %cst_1 = const.Declare tensor<2x4xf16> = dense<1.000000e+00> : tensor<2x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x2x1xf16>, tensor<2x3x10xf16>, tensor<2x3x1xf16>, tensor<2x4xf16> -> tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<2x3x10xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x10xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<2x3x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<2x4xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x4xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]])
    // CHECK-SAME:          tilingStrategy = [2, 2, 2, 1]
    // CHECK-SAME:     : tensor<2x100000x10xf16>, tensor<2x2x1xf16>, tensor<2x3x10xf16>, tensor<2x3x1xf16>, tensor<2x4xf16> -> tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>
}

// -----

// CHECK-LABEL:   func.func @GridSampleSplit
func.func @GridSampleSplit(%arg0: tensor<1x3x272x480xf16>, %arg1: tensor<1x272x480x2xf16>) -> tensor<1x3x272x480xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>
    return %0 : tensor<1x3x272x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GridSample(%arg0, %arg1)
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:     : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>

    // CHECK:       return [[OUTPUT]] :  tensor<1x3x272x480xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwGroupConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x256x96x96xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x8x3x3xf16>
func.func @SplitSwGroupConvOverOC(
        %input: tensor<1x256x96x96xf16>,
        %filter: tensor<256x8x3x3xf16>)
            -> tensor<1x256x96x96xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1],
        groups = 32 : i64

    } : tensor<1x256x96x96xf16>, tensor<256x8x3x3xf16> -> tensor<1x256x96x96xf16>
    return %1 : tensor<1x256x96x96xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 32, 1, 1]
    // CHECK-SAME:      -> tensor<1x256x96x96xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x96x96xf16>
}

// -----

func.func @SplitSwGroupConvOverH(
        %input: tensor<1x32x256x256xf16>,
        %filter: tensor<64x1x3x3xf16>)
            -> tensor<1x64x256x256xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1],
        groups = 32 : i64

    } : tensor<1x32x256x256xf16>, tensor<64x1x3x3xf16> -> tensor<1x64x256x256xf16>
    return %1 : tensor<1x64x256x256xf16>

    // CHECK-LABEL: func.func @SplitSwGroupConvOverH
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x256x256xf16>,
    // CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<64x1x3x3xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          groups = 32 : i64
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 10, 1]
    // CHECK-SAME:      -> tensor<1x64x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x64x256x256xf16>
}
