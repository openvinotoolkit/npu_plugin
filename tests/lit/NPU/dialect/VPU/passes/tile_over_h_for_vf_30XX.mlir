//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tile-over-h-for-vf %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE {
    IE.MemoryResource 2000000 bytes of @CMX_NN
    IE.ExecutorResource 2 of @DPU
}


// CHECK-LABEL: func.func @TOHConvConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x320x32x32xf16, {order = #NHWC}>
func.func @TOHConvConv(%input: tensor<1x320x32x32xf16, {order = #NHWC}>)
            -> tensor<1x320x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<320x320x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<320x320x3x3xf16, {order = #NHWC}>
    %cst_0 = const.Declare tensor<320x320x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<320x320x3x3xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>
    %cst_2 = const.Declare tensor<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input, %cst, %cst_2) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        rawFilterShape = [320, 320, 3, 3],
        strides = [2, 2],
        tilingStrategy = [1, 2, 1, 1]}
            -> tensor<1x320x16x16xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_0, %cst_1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [320, 320, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 2, 1, 1]}
            -> tensor<1x320x16x16xf16, {order = #NHWC}>
     return %1 : tensor<1x320x16x16xf16, {order = #NHWC}>

     // CHECK-DAG: [[WEIGHTS_0:%.+]] = const.Declare tensor<320x320x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x320x3x3xf16, {order = #NHWC}>
     // CHECK-DAG: [[WEIGHTS_1:%.+]] = const.Declare tensor<320x320x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x320x3x3xf16, {order = #NHWC}>
     // CHECK-DAG: [[WT_1:%.+]] = const.Declare tensor<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>
     // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>

     // CHECK:     [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_0]], [[WT_0]]) {
     // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
     // CHECK-SAME:     rawFilterShape = [320, 320, 3, 3], strides = [2, 2],
     // CHECK-SAME:     tilingStrategy = [1, 1, 4, 1]}
     // CHECK-SAME:         -> tensor<1x320x16x16xf16, {order = #NHWC}>

     // CHECK:      [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[WEIGHTS_1]], [[WT_1]]) {
     // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
     // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
     // CHECK-SAME:     rawFilterShape = [320, 320, 3, 3], strides = [1, 1],
     // CHECK-SAME:     tilingStrategy = [1, 1, 2, 1]}
     // CHECK-SAME:         -> tensor<1x320x16x16xf16, {order = #NHWC}>

     // CHECK:      return [[CONV_1]] : tensor<1x320x16x16xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotSegmentFaultForDepthToSpace
func.func @NotSegmentFaultForDepthToSpace(%input: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%input) {
        block_size = 2 : i64,
        mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        tilingStrategy = [1, 4, 1, 1]
        } : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    // CHECK:     [[D2S:%.+]] = VPU.DepthToSpace(%arg0) {
    // CHECK-SAME:     block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
    // CHECK-SAME:     tilingStrategy = [1, 1, 4, 1]
    // CHECK-SAME:     } : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    // CHECK-SAME:         -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    // CHECK:     return [[D2S]] : tensor<1x16x256x256x!qElemType, {order = #NHWC}>
}
