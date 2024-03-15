//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=false" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @BuildSubgraphEltwise(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
        ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_3 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
        ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2) 
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [26045], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>} 
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }

    return %2 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>


    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>, %cst_1 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], %arg4, %arg3) 
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV0]], [[CONV1]]) 
    //CHECK:        VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @NotBuildNotTiledSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK:      VPU.Yield [[CONV0]] 

    //CHECK:      [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst_1 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK:      VPU.Yield [[CONV1]] 
    
    //CHECK: return [[VERTICAL_FUSION1]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @NotBuildDiffMCSSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering> 
    //CHECK:      VPU.Yield [[CONV0]] 

    //CHECK:      [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst_1 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight> 
    //CHECK:      VPU.Yield [[CONV1]] 
    
    //CHECK: return [[VERTICAL_FUSION1]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @NotBuildTooLargeSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      %3 = VPU.NCE.Convolution(%2, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      %4 = VPU.NCE.Eltwise(%2, %3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [26045], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>} 
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK:      VPU.Yield [[CONV0]] 

    //CHECK:      [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst_1 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)  
    //CHECK:      [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV1]], %arg2, %arg3) 
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], [[CONV2]])  
    //CHECK:      VPU.Yield [[ELTWISE]]
    
    //CHECK: return [[VERTICAL_FUSION1]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}  

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128}>

func.func @BuildLargeSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x16x1x1x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_4 as %arg2: tensor<16x16x1x1x!qElemType2, {order = #NHWC}>, %cst_5 as %arg3: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %3 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>, %cst_2 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      %3 = VPU.NCE.Convolution(%2, %arg4, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      %4 = VPU.NCE.Eltwise(%2, %3) 
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [26045], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>} 
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst_2 as %arg2: tensor<16x16x1x1x!qElemType2, {order = #NHWC}>, %cst_3 as %arg3: tensor<16x1x1x4xsi32>, %cst as %arg4: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, 
    //CHECK-SAME:                         %cst_0 as %arg5: tensor<32x1x1x4xsi32>, %cst_1 as %arg6: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) 
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], %arg4, %arg5) 
    //CHECK:      [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV1]], %arg6, %arg5) 
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], [[CONV2]]) 
    //CHECK:      VPU.Yield [[ELTWISE]]
    
    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}   

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @NotBuildSubgraphOutOfSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> (tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <MAXIMUM>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_3 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>, 
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }
    %3 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %2 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg1, %arg2) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [26045], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>} 
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }

    return %1, %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>


    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK: VPU.Yield [[CONV]] 
    
    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK: VPU.Yield [[CONV]] 
    
    //CHECK: [[VERTICAL_FUSION2:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_1 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)  
    //CHECK: VPU.Yield [[CONV]] 
    
    //CHECK: [[VERTICAL_FUSION3:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, [[VERTICAL_FUSION2]] as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK: [[CONV:%.+]] = VPU.NCE.Eltwise(%arg1, %arg2)  
    //CHECK: VPU.Yield [[CONV]] 
    
    //CHECK: return [[VERTICAL_FUSION1]], [[VERTICAL_FUSION3]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}


// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>
!qElemType2 = !quant.uniform<u8:f16, 0.013744638480392158:128>

func.func @BuildSubgraphVFInput(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %arg1: tensor<1x16x256x256x!qElemType2, {order = #NHWC}>) -> (tensor<1x32x256x256x!qElemType, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg1 as %arg2: tensor<1x16x256x256x!qElemType2, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg4: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType2, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64, mode = <MAXIMUM>>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType2, {order = #NHWC}> 
      VPU.Yield %4 
    }               
    %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x32x256x256x!qElemType2, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg4: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
         ppe = #VPU.PPETask<clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64, mode = <LPRELU>>, 
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }
    %3 = VPU.VerticalFusion (%2 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %1 as %arg3: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg2, %arg3) 
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
         ppe = #VPU.PPETask<clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [26045], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = <NOOP>, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>} 
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> 
      VPU.Yield %4 
    }

    return %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion 
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4) 
    //CHECK: VPU.Yield [[CONV0]] 

    //CHECK: [[QUANTCAST:%.+]] = VPU.QuantizeCast([[VERTICAL_FUSION0]]) {dstElemType = !qElemType} 
    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion 
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4) 
    //CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], %arg5) 
    //CHECK: VPU.Yield [[ELTWISE]] 

    //CHECK: return [[VERTICAL_FUSION1]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotBuildLargeWeights(%arg0: tensor<1x256x26x26xf16, {order = #NHWC}>) -> tensor<1x256x26x26xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %cst_1 = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
        %cst_1 as %arg2: tensor<512x256x3x3xf16, {order = #NHWC}>,
        %cst_2 as %arg3: tensor<512x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x512x26x26xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad =  #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [512, 256, 3, 3], strides = [1, 1]} -> tensor<1x512x26x26xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x512x26x26xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x512x3x3xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x256x26x26xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [256, 512, 3, 3], strides = [1, 1]} -> tensor<1x256x26x26xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %1 : tensor<1x256x26x26xf16, {order = #NHWC}>

    //CHECK: [[VF_0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst_1 as %arg2: tensor<512x256x3x3xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst_2 as %arg3: tensor<512x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x512x26x26xf16, {order = #NHWC}>
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CHECK: [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as %arg1: tensor<1x512x26x26xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst as %arg2: tensor<256x512x3x3xf16, {order = #NHWC}>,
    //CHECK-SAME: %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x256x26x26xf16, {order = #NHWC}>
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)

    //CHECK: return [[VF_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildSubgraphConvSwishGroupConvVF(%arg0: tensor<1x16x176x176xf16, {order = #NHWC}>) -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %cst_2 = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %cst_4 = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.VerticalFusion (
          %arg0 as %arg1: tensor<1x16x176x176xf16, {order = #NHWC}>,
          %cst_0 as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
          %cst_1 as %arg3: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 3, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>,
          clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
          fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (
         %0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %3 = VPU.Swish(%arg1)
         {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
         tensor<1x96x176x176xf16, {order = #NHWC}>
            -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (
       %1 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>,
       %cst_2 as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
       %cst_3 as %arg3: tensor<96x1x1x4xsi32>,
       %cst_4 as %arg4: tensor<1x1x1x16xui8>) attributes {tilingStrategy = [1, 1, 4, 1]}
          -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %4 = VPU.NCE.DepthConvolution(
       %arg1, %arg2, %arg3, %arg4) {activation_window_channel_length = 4 : i64,
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
       ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64,
       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
       rawFilterShape = [96, 1, 1, 1],
       strides = [1, 1]
       } -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %4
    }

    return %2 : tensor<1x96x176x176xf16, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x176x176xf16, {order = #NHWC}>,
    //CHECK-SAME:                        %cst as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:                        %cst_0 as %arg3: tensor<96x1x1x4xsi32>,
    //CHECK-SAME:                        %cst_1 as %arg4: tensor<1x1x1x16xui8>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[SWISH0:%.+]] = VPU.Swish([[CONV0]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], %arg2, %arg3, %arg4 ) {activation_window_channel_length = 4 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:        VPU.Yield [[DWCONV0]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x96x176x176xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildMultiplySoftmaxSubgraph(%arg0: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1: tensor<1x4x160x160xf16, {order = #NHWC}>) -> tensor<1x4x160x160xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
      %3 = VPU.Multiply(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x4x160x160xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
    //CHECK: [[MUL:%.+]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
    //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[MUL]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
    //CHECK:  VPU.Yield [[SOFTMAX]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x4x160x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @BuildSubgraphEltwiseWithViewLikeOpInput(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %arg1: tensor<1x16x512x256x!qElemType>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x512x256x!qElemType> -> tensor<1x16x512x256x!qElemType, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 256, 256]} inputs(%0 : tensor<1x16x512x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    %2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg4: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3, %arg4)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 0.2998046875 : f64, lrelu_mult = 1228 : i64, lrelu_shift = 12 : i64>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    %3 = VPU.VerticalFusion (%1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %2 as %arg3: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg2, %arg3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPETask<mode = <NOOP>, clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24118], in2_quant_mult = [26045], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [25869], quant_post_shift = 0 : i64, quant_shift = [30]>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHT:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reorder<#NHWC>]
    //CHECK-DAG: [[BIAS:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    //CHECK: [[LAYOUTCAST:%.+]] = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x512x256x!qElemType> -> tensor<1x16x512x256x!qElemType, {order = #NHWC}>
    //CHECK: [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 256, 256]} inputs(%0 : tensor<1x16x512x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:              [[WEIGHT]] as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:              [[BIAS]] as %arg4: tensor<32x1x1x4xsi32>,
    //CHECK-SAME:              [[SHAPECAST]] as %arg5: tensor<1x32x256x256x!qElemType, {order = #NHWC}>)
    //CHECK-SAME:              attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution(%arg2, %arg3, %arg4)
    //CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg5, [[CONV]])
    //CHECK:   VPU.Yield [[ELTWISE]]
    //CHECK:   return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}
