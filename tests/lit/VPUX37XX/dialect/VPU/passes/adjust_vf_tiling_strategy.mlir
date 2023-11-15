//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --adjust-vf-tiling-strategy %s | FileCheck %s


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @VFIncreaseTileStrategy1(%arg0: tensor<1x48x256x16xf16, {order = #NHWC}>) -> tensor<1x1024x256x16xf16, {order = #NHWC}> {
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
    // CHECK-SAME: tilingStrategy = [1, 1, 23, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.003883181594488189:127,0.0031930517962598425:127,0.0036140501968503938:127,0.0036563422736220473:127,0.0035063976377952754:127,0.0039908341535433069:127,0.0036659541092519685:127,0.003196896530511811:127,0.0035217765748031494:127,0.0032622570127952754:127,0.0038408895177165355:127,0.0035256213090551179:127,0.0038332000492125986:127,0.003371831938976378:127,0.0035813699557086616:127,0.0037024790846456692:127,0.0038197434793307088:127,0.0036121278297244095:127,0.0033449187992125986:127,0.0031161571112204725:127,0.0036505751722440945:127,0.0034890963336614172:127,0.0038735697588582678:127,0.0033756766732283465:127,0.0030584860974409451:127,0.0037178580216535432:127,0.003456416092519685:127,0.0033256951279527561:127,0.0033487635334645671:127,0.0041484682578740153:127,0.0041215551181102358:127,0.0034910187007874014:127}>

func.func @DoNotAdjustTiling(%arg0: tensor<1x16x256x256x!qElemType0, {order = #NHWC}>, %wt: tensor<32x1x1x4xsi32>, %weights_1: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %weights_2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType0, {order = #NHWC}>  {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType0, {order = #NHWC}>, %weights_1 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %wt as %arg3: tensor<32x1x1x4xsi32>, %weights_2 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType0, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64, fp_prelu_alpha = 0.2998046875 : f64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType0, {order = #NHWC}> 
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64, fp_prelu_alpha = 0.2998046875 : f64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType0, {order = #NHWC}> 
      %3 = VPU.NCE.Eltwise(%1, %2) 
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
         ppe = #VPU.PPETask<clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [25852], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>} 
         -> tensor<1x32x256x256x!qElemType0, {order = #NHWC}> 
      VPU.Yield %3 
    }

    return %0 : tensor<1x32x256x256x!qElemType0, {order = #NHWC}> 

    // CHECK: VPU.VerticalFusion
    // CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
}

