//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MergeNonTiledRegion(
                %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
                %arg1: tensor<80x48x1x1xf16, {order = #NHWC}>,
                %arg2: tensor<48x80x1x1xf16, {order = #NHWC}>)
                    -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %9 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4xf16, {order = #NHWC}>,
        %arg1 as %arg4: tensor<80x48x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg5: tensor<80x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}> {
      %12 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
      rawFilterShape = [80, 48, 1, 1], strides = [1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}>
      VPU.Yield %12
   }

    %10 = VPU.VerticalFusion (%9 as %arg3: tensor<1x80x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x80x1024x4xf16, {order = #NHWC}> {
      %12 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x1024x4xf16, {order = #NHWC}> -> tensor<1x80x1024x4xf16, {order = #NHWC}>
      VPU.Yield %12
   }

   %11 = VPU.VerticalFusion (%10 as %arg3: tensor<1x80x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg4: tensor<48x80x1x1xf16, {order = #NHWC}>,
        %cst as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %12 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
      rawFilterShape = [48, 80, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %12
   }

   return %11: tensor<1x48x1024x4xf16, {order = #NHWC}>

   // Merge non-tiled operations when the pipelining is enabled
   //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
   //CHECK-SAME:    tilingStrategy = [1, 1, 1, 1]
   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]])
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]], %arg6, %arg7)
   //CHECK: VPU.Yield [[CONV1]]

   //CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x1024x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.035155834871179917:128>
!qElemType1 = !quant.uniform<u8:f16, 0.011608168658088235:128>

func.func @MergeEltwiseVFRegions(
        %input0: tensor<1x128x1x8x!qElemType, {order = #NHWC}>,
        %input1: tensor<1x128x1x8x!qElemType1, {order = #NHWC}>)
                -> tensor<1x128x1x8xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (
        %input1 as %arg0: tensor<1x128x1x8x!qElemType1, {order = #NHWC}>,
        %input1 as %arg1: tensor<1x128x1x8x!qElemType1, {order = #NHWC}>)
         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x1x8xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Eltwise(%arg1, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_shift = [35], quant_post_shift = 0 : i64, in1_quant_mult = [24344], in2_quant_mult = [24344], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x128x1x8xf16, {order = #NHWC}>
        VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (
        %input0 as %arg0: tensor<1x128x1x8x!qElemType, {order = #NHWC}>,
        %input0 as %arg1: tensor<1x128x1x8x!qElemType, {order = #NHWC}>)
            attributes {tilingStrategy = [1, 1, 1, 1]}
        -> tensor<1x128x1x8xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Eltwise(%arg1, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_shift = [33], quant_post_shift = 0 : i64, in1_quant_mult = [18431], in2_quant_mult = [18431], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x128x1x8xf16, {order = #NHWC}>
        VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (
        %0 as %arg0: tensor<1x128x1x8xf16, {order = #NHWC}>,
        %1 as %arg1: tensor<1x128x1x8xf16, {order = #NHWC}>)
            attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x1x8xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Eltwise(%arg0, %arg1) {index = 91 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>} -> tensor<1x128x1x8xf16, {order = #NHWC}>
        VPU.Yield %3
    }
    return %2 : tensor<1x128x1x8xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:              %arg0 as %arg2: tensor<1x128x1x8x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:              %arg0 as %arg3: tensor<1x128x1x8x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:              %arg1 as %arg4: tensor<1x128x1x8x!qElemType1, {order = #NHWC}>
    //CHECK:   [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise(%arg3, %arg3)
    //CHECK-NOT: VPU.VerticalFusion
    //CHECK:   [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise(%arg4, %arg4)
    //CHECK-NOT: VPU.VerticalFusion
    //CHECK:   [[ELTWISE_2:%.+]] = VPU.NCE.Eltwise([[ELTWISE_1]], [[ELTWISE_0]])
    //CHECK:   VPU.Yield [[ELTWISE_2]]
    //CHECK:   return [[VERTICAL_FUSION]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 1.7717625636680454:133>
!qElemType1 = !quant.uniform<u8:f16, 0.21032495685652192:133>
!qElemType2 = !quant.uniform<u8:f16, 0.29408995684455425:127>
!qElemType3 = !quant.uniform<u8:f16, 0.37756476308785231:124>
!qElemType4 = !quant.uniform<u8<0:254>:f16:0, {
    6.6940949892434546E-4:127, 8.7403148178040511E-4:127, 2.9108136540322791E-4:127, 3.7283236234206854E-4:127,
     9.688830047141849E-4:127, 7.6691806316375732E-4:127, 9.1997604435823093E-4:127,  5.713524663542199E-4:127,
    5.1553066321245332E-4:127, 9.3398213855863557E-4:127, 7.5581441010077164E-4:127,  5.7886159560811801E-4:127,
    4.8511795054270528E-4:127, 9.5655563778764614E-4:127, 4.9497322069378346E-4:127,  4.9018079605628187E-4:127,
    5.8581539261059502E-4:127, 3.9637836766993906E-4:127, 9.1732492831748306E-4:127,  7.8142275960426636E-4:127,
    5.3712262178030542E-4:127, 3.8713369313187488E-4:127, 4.6372607233017448E-4:127,   8.419075115459172E-4:127,
    7.2534986602978448E-4:127, 6.9180471221293052E-4:127,  8.888495132679076E-4:127,  7.0475693058779862E-4:127,
    5.7530631934563949E-4:127, 7.5183192810674348E-4:127, 8.9201677267945658E-4:127,  9.4348814074448717E-4:127,
    7.3688887939678401E-4:127, 3.3150894904699854E-4:127, 3.3287161330538471E-4:127,  0.0010755761401859793:127,
    0.0010332077976286881:127, 2.9396928670838126E-4:127, 3.9299642007181962E-4:127,  9.7349978338076375E-4:127,
    4.6915619626758604E-4:127, 0.0010792270420104499:127, 0.0010847422316318423:127,  4.6119451757491105E-4:127,
    8.7086193439528696E-4:127, 6.0612145136660473E-4:127, 9.0578497629466021E-4:127,  0.0010480731725692749:127,
    6.7316247956959285E-4:127, 9.9519951137032087E-4:127, 7.8166076752144516E-4:127,  6.4673550485625983E-4:127,
    8.9507303603990806E-4:127, 7.7278747802644269E-4:127, 7.2932988405227661E-4:127,  7.5017801654620429E-4:127,
     8.264777697916106E-4:127, 9.6631865567109714E-4:127,  7.441232171584302E-4:127,  0.0010090753084092629:127,
    8.5596493848665489E-4:127, 4.4993985825636258E-4:127, 7.2590226498175795E-4:127,  8.4447872450971228E-4:127
}>

func.func @BuildSubgraphWithTwoBranchesOp(%arg0: tensor<1x64x64x192xf16, {order = #NHWC}>, %arg1: tensor<1x64x64x192xf16, {order = #NHWC}>) -> tensor<1x64x64x192x!qElemType3, {order = #NHWC}> {
    %cst = const.Declare tensor<64x64x3x3x!qElemType4, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType4>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg4: tensor<1x64x64x192xf16, {order = #NHWC}>, %arg0 as %arg5: tensor<1x64x64x192xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x64x192x!qElemType0, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg4, %arg5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <LPRELU>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1311 : i64, lrelu_shift = 17 : i64, quant_scale = [0.56440971296386322], fp_prelu_alpha = 0.005645302589982748 : f64>} -> tensor<1x64x64x192x!qElemType0, {order = #NHWC}> 
      VPU.Yield %4
    }
    %1 = VPU.VerticalFusion (%0 as %arg4: tensor<1x64x64x192x!qElemType0, {order = #NHWC}>, %cst as %arg5: tensor<64x64x3x3x!qElemType4, {order = #NHWC}>, %cst_0 as %arg6: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x64x192x!qElemType1, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} -> tensor<1x64x64x192x!qElemType1, {order = #NHWC}> 
      VPU.Yield %4 
    }
    %2 = VPU.VerticalFusion (%arg1 as %arg4: tensor<1x64x64x192xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x64x192x!qElemType2, {order = #NHWC}> {
      %4 = VPU.NCE.AveragePool(%arg4) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [3.4003201290160523], fp_prelu_alpha = 3.400320291519165 : f64>, strides = [1, 1]} -> tensor<1x64x64x192x!qElemType2, {order = #NHWC}> 
      VPU.Yield %4 
    }
    %3 = VPU.VerticalFusion (%1 as %arg4: tensor<1x64x64x192x!qElemType1, {order = #NHWC}>, %2 as %arg5: tensor<1x64x64x192x!qElemType2, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x64x192x!qElemType3, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg4, %arg5) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPETask<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [21696], quant_shift = [30], quant_post_shift = 0 : i64, in1_quant_mult = [27567], in2_quant_mult = [38546], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x64x64x192x!qElemType3, {order = #NHWC}> 
      VPU.Yield %4 
    }

    return %3: tensor<1x64x64x192x!qElemType3, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion 
    //CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    //CHECK: VPU.NCE.AveragePool 
    //CHECK: VPU.NCE.Eltwise
    //CHECK: VPU.NCE.Convolution 
    //CHECK: [[ELT:%.+]] = VPU.NCE.Eltwise 
    //CHECK: VPU.Yield [[ELT]]
    
    //CHECK: return [[VERTICAL_FUSION]]
}
