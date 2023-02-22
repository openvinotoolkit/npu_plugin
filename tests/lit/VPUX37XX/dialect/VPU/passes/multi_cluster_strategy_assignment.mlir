//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOH
func @ConvAssignedSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
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
func @ConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
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
func @ConvAssignedSOK(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1x1xf16, {order = #NHWC}> {
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
func @DepthConvAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
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
func @DepthConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
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
func @DepthConvAssignedSOK(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
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
func @MaxPoolAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
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
func @MaxPoolAssignedClustering(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
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
func @EltwiseAddAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
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
func @ConvAssignedStrategyForLargeLayer(%arg0: tensor<1x64x608x608xf16, {order = #NHWC}>) -> tensor<1x80x608x608xf16, {order = #NHWC}> {
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
func @SparseConvAssignedSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 01 : i64},
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    // CHECK:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [80, 64, 3, 3],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOK
func @SparseConvAssignedSOK(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1024x1x1x128xi1> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<1024x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1024x1x1x128xi1>, is_weights>

    %weights_table = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [1024, 128, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1024x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1024x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1024x1x1x128xi1>, is_weights>

    // CHECK:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "SplitOverKernel",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [1024, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>>
}

// -----

// 1008 channels does not divide equally across the two clusters while keeping the channels aligned, making the SOK strategy illegal

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvUnequalChannelsAssignedClustering
func @SparseConvUnequalChannelsAssignedClustering(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1008x1x1x128xi1>, is_weights>

    %weights_table = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1008x1x1x128xi1>, is_weights>

    // CHECK:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "Clustering",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [1008, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedClustering
func @TanhAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %1 = VPU.Tanh(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0) {multiClusterStrategy = "Clustering"} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultTANH]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedClustering
func @MVNAssignedClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = "Clustering", normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedSplitOverKernel
func @MVNAssignedSplitOverKernel(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = "SplitOverKernel", normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TanhAfterConvAssignedClustering
func @TanhAfterConvAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

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

