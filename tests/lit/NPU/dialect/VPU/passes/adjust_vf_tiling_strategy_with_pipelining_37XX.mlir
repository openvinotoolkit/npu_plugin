//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-vf-tiling-strategy="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DoNotIncreaseWrongPattern(%arg0: tensor<1x48x256x16xf16, {order = #NHWC}>) -> tensor<1x1024x256x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x48x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1024x48x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x48x256x16xf16, {order = #NHWC}>, %cst as %arg3: tensor<1024x48x1x1xf16, {order = #NHWC}>, 
    %cst_0 as %arg4: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 5, 1]} 
        -> tensor<1x1024x256x16xf16, {order = #NHWC}> { 
        %1 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) 
          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
          ppe = #VPU.PPETask<clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, 
          fp_prelu_alpha =   1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>>, 
          rawFilterShape = [1024, 48, 1, 1], strides = [1, 1]} 
        -> tensor<1x1024x256x16xf16, {order = #NHWC}> 
        %2 = VPU.SoftMax(%1) 
          {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1024x256x16xf16, {order = #NHWC}> 
           -> tensor<1x1024x256x16xf16, {order = #NHWC}> 
        VPU.Yield %2 
    }
    return %0 : tensor<1x1024x256x16xf16, {order = #NHWC}>

    // The tiling strategy is the same as vf pipelining disabled when the pattern doesn't match DPU-SW-DPU
    // CHECK: VPU.VerticalFusion
    // CHECK-SAME: tilingStrategy = [1, 1, 8, 1]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16, 0.00565029593075023:128>


func.func @VFIncreaseTileStrategy2(%arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
%arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_2 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    
    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>, %arg1 as %arg4: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>, %arg1 as %arg6: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst_2 as %arg7: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 22, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> 
   { %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) 
       {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
        rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> 
     %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}> 
     %3 = VPU.NCE.Convolution(%2, %arg6, %arg7) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
    rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> 
VPU.Yield %3 }    

    return %0 : tensor<1x48x1024x4xf16, {order = #NHWC}>

    // CHECK: VPU.VerticalFusion
    // CHECK-SAME: tilingStrategy = [1, 1, 64, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @VFIncreaseTileStrategyForNonTiledRegion(%arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1: tensor<80x48x1x1xf16, {order = #NHWC}>, %arg2: tensor<48x80x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>
    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<80x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<80x1x1x4xsi32>, %arg2 as %arg6: tensor<48x80x1x1xf16, {order = #NHWC}>, %cst as %arg7: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [80, 48, 1, 1], strides = [1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}>
        %2 = VPU.SoftMax(%1) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x1024x4xf16, {order = #NHWC}> -> tensor<1x80x1024x4xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %arg6, %arg7) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [48, 80, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
        VPU.Yield %3
    }
    return %0 : tensor<1x48x1024x4xf16, {order = #NHWC}>

   // Don't unroll the non-tiled VF subgraph when the pipelining is enabled
   //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
   //CHECK-SAME:    tilingStrategy = [1, 1, 2, 1]
   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]])
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], %arg6, %arg7)
   //CHECK: VPU.Yield [[CONV1]]

   //CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x1024x4xf16, {order = #NHWC}>
}
