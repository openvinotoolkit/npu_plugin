//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildOutCMXSubgraph(%arg0: tensor<1x32x150x256xf16, {order = #NHWC}>) -> tensor<1x32x150x256xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x32x11x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x32x11x1xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x150x256xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<16x32x11x1xf16, {order = #NHWC}>, %cst_5 as %arg3: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x150x256xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 5 : i64, bottom = 5 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64, fp_prelu_alpha = 0.2998046875 : f64>, 
         rawFilterShape = [16, 32, 11, 1], strides = [1, 1]} -> tensor<1x16x150x256xf16, {order = #NHWC}> 
      VPU.Yield %2 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x150x256xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x150x256xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64, fp_prelu_alpha = 0.2998046875 : f64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x150x256xf16, {order = #NHWC}> 
      VPU.Yield %2
    }
    return %1 : tensor<1x32x150x256xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x150x256xf16, {order = #NHWC}>, %cst_1 as %arg2: tensor<16x32x11x1xf16, {order = #NHWC}>, %cst_2 as %arg3: tensor<16x1x1x4xsi32>, %cst as %arg4: tensor<32x16x3x3xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x32x150x256xf16, {order = #NHWC}> {
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], %arg4, %arg5)  
    //CHECK:  VPU.Yield [[CONV1]]
    //CHECK: return [[VERTICAL_FUSION]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @BuildSubgraphNotConstInput(%arg0: tensor<1x48x256x16xf16, {order = #NHWC}>, %arg1: tensor<8x4096x40xf16>, %arg2: tensor<8x4096x40xf16>) -> tensor<1x48x256x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>

    %0 = VPU.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [8, 4096, 40, 1]} : tensor<8x4096x40xf16> -> tensor<8x4096x40x1xf16> 
    %1 = VPU.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHCW, tilingStrategy = [1, 1, 3, 1]} : tensor<8x4096x40x1xf16> -> tensor<8x40x4096x1xf16> 
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [320, 4096, 1, 1]} : tensor<8x40x4096x1xf16> -> tensor<320x4096x1x1xf16> 
    %3 = VPU.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<320x4096x1x1xf16> -> tensor<320x4096x1x1xf16, {order = #NHWC}>
    %4 = VPU.Slice %3 [0, 0, 0, 0] [48, 4096, 1, 1] : tensor<320x4096x1x1xf16, {order = #NHWC}> to tensor<48x4096x1x1xf16, {order = #NHWC}>
    %5 = VPU.AffineReshape(%arg2) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32768, 40, 1, 1]} : tensor<8x4096x40xf16> -> tensor<32768x40x1x1xf16> 
    %6 = VPU.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<32768x40x1x1xf16> -> tensor<32768x40x1x1xf16, {order = #NHWC}> 
    %7 = VPU.Expand(%6) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<32768x40x1x1xf16, {order = #NHWC}> -> tensor<32768x48x1x1xf16, {order = #NHWC}>
    %8 = VPU.Slice %7 [0, 0, 0, 0] [4096, 48, 1, 1] : tensor<32768x48x1x1xf16, {order = #NHWC}> to tensor<4096x48x1x1xf16, {order = #NHWC}>
    %9 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x256x16xf16, {order = #NHWC}>, %8 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x4096x256x16xf16, {order = #NHWC}> { 
      %11 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) 
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
      rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x256x16xf16, {order = #NHWC}> 
      %12 = VPU.SoftMax(%11) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x256x16xf16, {order = #NHWC}> -> tensor<1x4096x256x16xf16, {order = #NHWC}> 
      VPU.Yield %12
   } 
   
   %10 = VPU.VerticalFusion (%9 as %arg3: tensor<1x4096x256x16xf16, {order = #NHWC}>, %4 as %arg4: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 22, 1]} -> tensor<1x48x256x16xf16, {order = #NHWC}> { 
      %11 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) 
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
      rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x256x16xf16, {order = #NHWC}> 
      VPU.Yield %11
   }

   return %10: tensor<1x48x256x16xf16, {order = #NHWC}>

   //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion 
   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]]) 
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], %arg6, %arg7) 
   //CHECK: VPU.Yield [[CONV1]] 

   //CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x256x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @PartialBuildWithWeightsConstraints(%arg0: tensor<1x256x26x26xf16, {order = #NHWC}>) -> tensor<1x256x26x26xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %cst_1 = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
        %cst_1 as %arg2: tensor<256x256x3x3xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x256x26x26xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x26x26xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]}
            -> tensor<1x256x26x26xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} -> tensor<1x256x26x26xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (
        %1 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x256x26x26xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} -> tensor<1x256x26x26xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %2 : tensor<1x256x26x26xf16, {order = #NHWC}>

    // %1 and %2 are merged. %0 is not merged to avoid too large weights size
    //CHECK: [[VF_0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst_1 as %arg2: tensor<256x256x3x3xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x256x26x26xf16, {order = #NHWC}>
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CHECK: [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x256x26x26xf16, {order = #NHWC}>
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK:    [[CONV_2:%.+]] = VPU.NCE.Convolution(%2, %arg2, %arg3)

    //CHECK: return [[VF_1]]
}
