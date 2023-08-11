//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType0 = !quant.uniform<u8:f16, 0.0017310915969488189:127>

// CHECK-LABEL: @AdaptSWAlignmentToAvoidSpilling
func.func @AdaptSWAlignmentToAvoidSpilling(%arg0: tensor<1x256x16x32xf16, {order = #NHWC}>) -> tensor<1x256x16x32xf16, {order = #NHWC}> {
    %cst_1 = const.Declare tensor<256x256x3x3x!qElemType0, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16, {order = #NHWC}>
    %cst_2 = const.Declare tensor<256x256x3x3x!qElemType0, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16, {order = #NHWC}>
    %cst_3 = const.Declare tensor<256x256x3x3x!qElemType0, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16, {order = #NHWC}>
    %cst_13 = const.Declare tensor<1x1x1x16xui8> = dense<[[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>
    %cst_18 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_37 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_38 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<10.0>: tensor<1x1x1x256xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[256, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[256, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_39 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_40 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_41 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<1x1x1x256xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[256, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[256, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_42 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_43 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_44 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<1x1x1x256xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[256, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[256, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_45 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>

    %12 = VPU.MVN(%arg0) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %13 = VPU.NCE.DepthConvolution(%12, %cst_44, %cst_45, %cst_13) {activation_window_channel_length = 4 : i64, pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 26.11199951171875 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [256, 1, 1, 1], strides = [1, 1]} -> tensor<1x256x16x32x!qElemType0, {order = #NHWC}>
    %14 = VPU.NCE.Convolution(%13, %cst_1, %cst_43) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %15 = VPU.MVN(%14) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %16 = VPU.NCE.DepthConvolution(%15, %cst_41, %cst_42, %cst_13) {activation_window_channel_length = 4 : i64, pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 23.842220306396484 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [256, 1, 1, 1], strides = [1, 1]} -> tensor<1x256x16x32x!qElemType0, {order = #NHWC}>
    %17 = VPU.NCE.Convolution(%16, %cst_2, %cst_40) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %18 = VPU.MVN(%17) {across_channels = false, eps = 6.0892105102539063E-4 : f64, normalize_variance = true} : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %19 = VPU.NCE.DepthConvolution(%18, %cst_38, %cst_39, %cst_13) {activation_window_channel_length = 4 : i64, pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 16.202531814575195 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [256, 1, 1, 1], strides = [1, 1]} -> tensor<1x256x16x32x!qElemType0, {order = #NHWC}>
    %20 = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs(%13 : tensor<1x256x16x32x!qElemType0, {order = #NHWC}>) -> tensor<1x16x32x256x!qElemType0, {order = #NHWC}>
    %21 = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs(%19 : tensor<1x256x16x32x!qElemType0, {order = #NHWC}>) -> tensor<1x16x32x256x!qElemType0, {order = #NHWC}>
    %22 = VPU.NCE.Eltwise(%20, %21) {op_type = "ADD", ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [20078], in2_quant_mult = [32358], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_mult = [22783], quant_post_shift = 0 : i64, quant_shift = [30]}} -> tensor<1x16x32x256x!qElemType0, {order = #NHWC}>
    %23 = VPU.ShapeCast {shape = [1, 256, 16, 32]} inputs(%22 : tensor<1x16x32x256x!qElemType0, {order = #NHWC}>) -> tensor<1x256x16x32x!qElemType0, {order = #NHWC}>
    %24 = VPU.NCE.Convolution(%23, %cst_3, %cst_37) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} -> tensor<1x256x16x32xf16, {order = #NHWC}>
    return %24 : tensor<1x256x16x32xf16, {order = #NHWC}>

    //CHECK-COUNT-14:   const.Declare
    //CHECK:            [[VAL0:%.+]] = VPU.MVN(%arg0)
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL1:%.+]] = VPU.NCE.DepthConvolution([[VAL0]],
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]],
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL3:%.+]] = VPU.MVN([[VAL2]])
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL3]]
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL5:%.+]] = VPU.NCE.Convolution([[VAL4]]
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL6:%.+]] = VPU.MVN([[VAL5]]
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL7:%.+]] = VPU.NCE.DepthConvolution([[VAL6]]
    //CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    //CHECK:            [[VAL8:%.+]] = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs([[VAL1]]
    //CHECK:            [[VAL9:%.+]] = VPU.ShapeCast {shape = [1, 16, 32, 256]} inputs([[VAL7]]
    //CHECK:            [[VAL10:%.+]] = VPU.NCE.Eltwise([[VAL8]], [[VAL9]])
    //CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    //CHECK:            [[VAL11:%.+]] = VPU.ShapeCast {shape = [1, 256, 16, 32]} inputs([[VAL10]]
    //CHECK:            [[VAL12:%.+]] = VPU.NCE.Convolution([[VAL11]],
    //CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    //CHECK:            return [[VAL12]] : tensor<1x256x16x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOH
func.func @ConvAssignedSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:    {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = "SplitOverKernel", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x48x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = "SplitOverKernel", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x48x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x48x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvAssignedSOH
func.func @DepthConvAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 18 : i64, multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 18 : i64, multiClusterStrategy = "SplitOverKernel", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x128x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 18 : i64, multiClusterStrategy = "SplitOverKernel", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOH
func.func @MaxPoolAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], multiClusterStrategy = "SplitOverHeight", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedClustering
func.func @MaxPoolAssignedClustering(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], multiClusterStrategy = "Clustering", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddAssignedSOH
func.func @EltwiseAddAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { op_type = "ADD" } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = "SplitOverHeight", op_type = "ADD"} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedStrategyForLargeLayer
func.func @ConvAssignedStrategyForLargeLayer(%arg0: tensor<1x64x608x608xf16, {order = #NHWC}>) -> tensor<1x80x608x608xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x608x608xf16, {order = #NHWC}>
    return %0 : tensor<1x80x608x608xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:    {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x608x608xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x608x608xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOH
func.func @SparseConvAssignedSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 01 : i64},
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>, is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.CompressionScheme

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [80, 64, 3, 3],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOK
func.func @SparseConvAssignedSOK(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:       {compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>, is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.CompressionScheme

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "SplitOverKernel",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [1008, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>
}

// -----

// Weights contain only sparse values, so the weight splits would have only sparse values as well
// SOK is avoided in case any split has empty weights

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvNotAssignedSOK
func.func @SparseConvNotAssignedSOK(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1008x1x1x128xi1> = dense<0.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {
            compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>, is_weights
        } -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1008x1x1x128xi1>,
                               is_weights, #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1008x1x1x128xi1> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {
    // CHECK-SAME:          compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>, is_weights
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                             sparsity_map=tensor<1008x1x1x128xi1>,
    // CHECK-SAME:                             is_weights, #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<0> : tensor<1008xi64>, alignment = 16 : i64>>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "Clustering",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [1008, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedClustering
func.func @TanhAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %1 = VPU.Tanh(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0) {multiClusterStrategy = "Clustering"} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultTANH]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedSOHForEqualCost
func.func @TanhAssignedSOHForEqualCost(%arg0: tensor<1x4x32x512xf16, {order = #NCHW}>) -> tensor<1x4x32x512xf16, {order = #NCHW}> {

    %1 = VPU.Tanh(%arg0) : tensor<1x4x32x512xf16, {order = #NCHW}> -> tensor<1x4x32x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x32x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0) {multiClusterStrategy = "SplitOverHeight"} : tensor<1x4x32x512xf16, {order = #NCHW}> -> tensor<1x4x32x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultTANH]] : tensor<1x4x32x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedClustering
func.func @MVNAssignedClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = "Clustering", normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedClustering
func.func @SoftMaxAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %1 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64, multiClusterStrategy = "Clustering"} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x1x512xf16, {order = #NCHW}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverKernel
func.func @SoftMaxAssignedSplitOverKernel(%arg0: tensor<1x4x2x512xf16, {order = #NCHW}>) -> tensor<1x4x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x4x2x512xf16, {order = #NCHW}> -> tensor<1x4x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = "SplitOverKernel"} : tensor<1x4x2x512xf16, {order = #NCHW}> -> tensor<1x4x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x2x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxOnChannelNotAssignedSplitOverKernel
func.func @SoftMaxOnChannelNotAssignedSplitOverKernel(%arg0: tensor<1x4x2x512xf16, {order = #NCHW}>) -> tensor<1x4x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x4x2x512xf16, {order = #NCHW}> -> tensor<1x4x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64, multiClusterStrategy = "Clustering"} : tensor<1x4x2x512xf16, {order = #NCHW}> -> tensor<1x4x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x2x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverHeight
func.func @SoftMaxAssignedSplitOverHeight(%arg0: tensor<1x1x2x512xf16, {order = #NCHW}>) -> tensor<1x1x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x1x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = "SplitOverHeight"} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x1x2x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxOnHeightNotAssignedSplitOverHeight
func.func @SoftMaxOnHeightNotAssignedSplitOverHeight(%arg0: tensor<1x1x2x512xf16, {order = #NCHW}>) -> tensor<1x1x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 2} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x1x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64, multiClusterStrategy = "Clustering"} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x1x2x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NHWCSoftMaxAssignedSplitOverHeight
func.func @NHWCSoftMaxAssignedSplitOverHeight(%arg0: tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    return %1 : tensor<1x8x4x76xf16, {order = #NHWC}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = "SplitOverHeight"} : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16, {order = #NHWC}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x8x4x76xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedSplitOverKernel
func.func @MVNAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = "SplitOverKernel", normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TanhAfterConvAssignedClustering
func.func @TanhAfterConvAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

    %39 = const.Declare tensor<1x1024x1x1xf16> = dense<10.1> : tensor<1x1024x1x1xf16>
    %40 = const.Declare tensor<1x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1x1xf16, {order = #NHWC}>

    %cst_7 = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1024xf16>, [#const.Reshape<[1024, 1024]>, #const.Reshape<[1024, 1024, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_27 = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>

    %42 = VPU.NCE.Convolution(%40, %cst_7, %cst_27) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    %45 = VPU.Tanh(%42) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    return %45 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:   [[ResultConvolution:%.*]] = VPU.NCE.Convolution([[cst0:%.*]], [[cst1:%.*]], [[cst2:%.*]]) {multiClusterStrategy = "SplitOverKernel", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    //CHECK:   [[ResultTanh:%.*]] = VPU.Tanh([[ResultConvolution]]) {multiClusterStrategy = "Clustering"} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    //CHECK:   return [[ResultTanh]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @InterpolateHalfPixelAssignedSOHOverlapped
func.func @InterpolateHalfPixelAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = "SplitOverHeightOverlapped", operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersAssignedSOHOverlapped
func.func @InterpolateAlignCornersAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = "SplitOverHeightOverlapped", operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersAssignedClustering
func.func @InterpolateAlignCornersAssignedClustering(%arg0: tensor<1x1x1x160xf16>) -> tensor<1x1x2x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [2, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x2x320xf16>
    return %0 : tensor<1x1x2x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = "Clustering", operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [2, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x2x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x2x320xf16>
}

// -----

// CHECK-LABEL: @InterpolatePytorchHalfPixelAssignedSOHOverlapped
func.func @InterpolatePytorchHalfPixelAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = "SplitOverHeightOverlapped", operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolatePytorchHalfPixelAssignedClustering
func.func @InterpolatePytorchHalfPixelAssignedClustering(%arg0: tensor<1x1x1x160xf16>) -> tensor<1x1x2x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [2, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x2x320xf16>
    return %0 : tensor<1x1x2x320xf16>
    //CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], multiClusterStrategy = "Clustering", operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [2, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x2x320xf16>
    //CHECK:   return [[INTERPOLATE]] : tensor<1x1x2x320xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwishAssignedSplitOverHeight
func.func @SwishAssignedSplitOverHeight(%arg0: tensor<1x1x64x64xf16, {order = #NHWC}>) -> tensor<1x1x64x64xf16, {order = #NHWC}> {

    %0 = VPU.Swish(%arg0) : tensor<1x1x64x64xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x1x64x64xf16, {order = #NHWC}>

    //CHECK:      [[ResultSwish:%.*]] = VPU.Swish(%arg0) {multiClusterStrategy = "SplitOverHeight"} :
    //CHECK-SAME: tensor<1x1x64x64xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>
    //CHECK:      return [[ResultSwish]] : tensor<1x1x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwishAssignedSplitOverKernel
func.func @SwishAssignedSplitOverKernel(%arg0: tensor<1x64x1x64xf16, {order = #NHWC}>) -> tensor<1x64x1x64xf16, {order = #NHWC}> {

    %0 = VPU.Swish(%arg0) : tensor<1x64x1x64xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>

    return %0 : tensor<1x64x1x64xf16, {order = #NHWC}>

    //CHECK:      [[ResultSwish:%.*]] = VPU.Swish(%arg0) {multiClusterStrategy = "SplitOverKernel"} :
    //CHECK-SAME: tensor<1x64x1x64xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHWC}>
    //CHECK:      return [[ResultSwish]] : tensor<1x64x1x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GeluAssignedSplitOverHeight
func.func @GeluAssignedSplitOverHeight(%arg0: tensor<1x1x4x512xf16, {order = #NCHW}>) -> tensor<1x1x4x512xf16, {order = #NCHW}> {

    %0 = VPU.Gelu(%arg0) : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x4x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultGelu:%.*]] = VPU.Gelu(%arg0) {multiClusterStrategy = "SplitOverHeight"} : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultGelu]] : tensor<1x1x4x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GeluAssignedSplitOverKernel
func.func @GeluAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.Gelu(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultGelu:%.*]] = VPU.Gelu(%arg0) {multiClusterStrategy = "SplitOverKernel"} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultGelu]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOHOverlapped
func.func @CompressConvolutionAssignedSOHOverlapped(%arg0: tensor<1x3x224x224xf16, {order = #NHWC}>) -> tensor<1x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.ConvertElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table)
            {
                cm_sp_pattern = 15 : i64,
                pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
                ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64,
                    lrelu_shift = 0 : i64, mode = "NOOP"},
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
            } -> tensor<1x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    //CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    //CHECK:        [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    //CHECK-SAME:           tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:           {cm_sp_pattern = 15 : i64, multiClusterStrategy = "SplitOverHeightOverlapped",
    //CHECK-SAME:           pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}
    //CHECK-SAME:           ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64,
    //CHECK-SAME:           lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"}
    //CHECK-SAME:           rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
    //CHECK-SAME:           -> tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedSplitOverHeight
func.func @MultiplyAssignedSplitOverHeight(%arg0: tensor<1x32x44x44xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x44x44xf16, {order = #NHWC}>) -> tensor<1x32x44x44xf16, {order = #NHWC}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x32x44x44xf16, {order = #NHWC}>,
                tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x44x44xf16, {order = #NHWC}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = "SplitOverHeight"} : tensor<1x32x44x44xf16, {order = #NHWC}>, tensor<1x1x44x44xf16, {order = #NHWC}> -> tensor<1x32x44x44xf16, {order = #NHWC}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x32x44x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedSplitOverKernel
func.func @MultiplyAssignedSplitOverKernel(%arg0: tensor<1x32x1x44xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x1x44xf16, {order = #NHWC}>) -> tensor<1x32x1x44xf16, {order = #NHWC}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x32x1x44xf16, {order = #NHWC}>,
                tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x44xf16, {order = #NHWC}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = "SplitOverKernel"} : tensor<1x32x1x44xf16, {order = #NHWC}>, tensor<1x1x1x44xf16, {order = #NHWC}> -> tensor<1x32x1x44xf16, {order = #NHWC}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x32x1x44xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedClustering
func.func @MultiplyAssignedClustering(%arg0: tensor<1x1x1x44xf16, {order = #NHWC}>,
            %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x44xf16, {order = #NHWC}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x1x44xf16, {order = #NHWC}>,
                tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x44xf16, {order = #NHWC}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = "Clustering"} : tensor<1x1x1x44xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x44xf16, {order = #NHWC}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x1x1x44xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @HSwishAssignedSplitOverHeight
func.func @HSwishAssignedSplitOverHeight(%arg0: tensor<1x1x4x512xf16, {order = #NCHW}>) -> tensor<1x1x4x512xf16, {order = #NCHW}> {

    %0 = VPU.HSwish(%arg0) : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x4x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultHSwish:%.*]] = VPU.HSwish(%arg0) {multiClusterStrategy = "SplitOverHeight"} : tensor<1x1x4x512xf16, {order = #NCHW}> -> tensor<1x1x4x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultHSwish]] : tensor<1x1x4x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @HSwishAssignedSplitOverKernel
func.func @HSwishAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.HSwish(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultHSwish:%.*]] = VPU.HSwish(%arg0) {multiClusterStrategy = "SplitOverKernel"} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultHSwish]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedClustering
func.func @InterpolateNearestAssignedClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = "NEAREST",
            nearest_mode = "FLOOR",
            coordinate_transformation_mode = "ASYMMETRIC",
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        mode = "NEAREST",
        scales_attr = [2, 2],
        ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "NOOP"}
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = "NEAREST",
    // CHECK-SAME:      multiClusterStrategy = "Clustering",
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// CHECK-LABEL: @InterpolateNearestAssignedSOK
func.func @InterpolateNearestAssignedSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = "NEAREST",
            nearest_mode = "FLOOR",
            coordinate_transformation_mode = "ASYMMETRIC",
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x10x20xi1>,
                           storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = "NEAREST", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 1, 1],
        mode = "NEAREST",
        scales_attr = [2, 2],
        ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "NOOP"}
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = "NEAREST",
    // CHECK-SAME:      multiClusterStrategy = "SplitOverKernel",
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:      rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2]}
    // CHECK-SAME:      -> tensor<1x64x10x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedClustering
func.func @InterpolateBilinearAssignedClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x3x3xi1> = dense<1> : tensor<1x16x3x3xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 3, 3]>
    } -> tensor<1x1x3x3xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = "BILINEAR",
            nearest_mode = "FLOOR",
            coordinate_transformation_mode = "ASYMMETRIC",
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 3, 3]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x3x3xi1>,
                           storage_element_table=tensor<1x1x3x3xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 3, 3]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 2, 2],
        mode = "BILINEAR",
        scales_attr = [2, 2],
        ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "NOOP"}
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x3x3xi1> = dense<true> : tensor<1x16x3x3xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = "BILINEAR",
    // CHECK-SAME:      multiClusterStrategy = "Clustering",
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:      rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedSOK
func.func @InterpolateBilinearAssignedSOK(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x11x21xi1> = dense<1> : tensor<1x64x11x21xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 64, dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 11, 21]>
    } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = "BILINEAR",
            nearest_mode = "FLOOR",
            coordinate_transformation_mode = "ASYMMETRIC",
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } -> !VPU.SparseTensor<data=tensor<1x64x5x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x11x21xi1>,
                           storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = "BILINEAR", nearest_mode = "FLOOR", coordinate_transformation_mode = "ASYMMETRIC",
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 64, 11, 21]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 2, 2],
        mode = "BILINEAR",
        scales_attr = [2, 2],
        ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "NOOP"}
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x64x11x21xi1> = dense<true> : tensor<1x64x11x21xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = "BILINEAR",
    // CHECK-SAME:      multiClusterStrategy = "SplitOverKernel",
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:      rawFilterShape = [64, 64, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2]}
    // CHECK-SAME:      -> tensor<1x64x10x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}
