//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s

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

// CHECK-LABEL: @ConvAssignedSOH
func.func @ConvAssignedSOH(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x1024x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} -> tensor<1x1024x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x16x16xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x1024x16x16xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x1024x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOH
func.func @ConvAssignedSOH(%arg0: tensor<1x64x14x14xf16, {order = #NHWC}>) -> tensor<1x48x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x48x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x48x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x32x3x3xf16, {order = #NHWC}>) -> tensor<1x64x3x3xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 32, 1, 1], strides = [1, 1]} -> tensor<1x64x3x3xf16, {order = #NHWC}>
    return %0 : tensor<1x64x3x3xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = "SplitOverKernel", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 32, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x64x3x3xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x64x3x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CMConvAssignedSOHOverlapped
func.func @CMConvAssignedSOHOverlapped(%arg0: tensor<1x3x224x224xf16, {order = #NCHW}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 1, 1, 27]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 81 : i64, pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [32, 3, 3, 3], strides = [2, 2]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x1x1x32xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 1, 1, 27]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]] )
    //CHECK-SAME:   {activation_window_channel_length = 81 : i64, multiClusterStrategy = "SplitOverHeightOverlapped", pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [32, 3, 3, 3], strides = [2, 2]}
    //CHECK-SAME:   -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CMConvAssignedSOHOverlapped
func.func @CMConvAssignedSOHOverlapped(%arg0: tensor<1x3x16x16xf16, {order = #NCHW}>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[48, 1, 1, 27]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 54 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 3, 3, 3], strides = [1, 1]} -> tensor<1x48x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x48x16x16xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x1x1x32xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<48x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[48, 1, 1, 27]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 5]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]] )
    //CHECK-SAME:   {activation_window_channel_length = 54 : i64, multiClusterStrategy = "SplitOverHeightOverlapped", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 3, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x48x16x16xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x48x16x16xf16, {order = #NHWC}>
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

// CHECK-LABEL: @DepthConvAssignedSOH
func.func @DepthConvAssignedSOH(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x128x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x128x16x16xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 18 : i64, multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x128x16x16xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x128x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x128x3x3xf16, {order = #NHWC}>) -> tensor<1x128x3x3xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x3x3xf16, {order = #NHWC}>
    return %0 : tensor<1x128x3x3xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 18 : i64, multiClusterStrategy = "SplitOverKernel", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x128x3x3xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x128x3x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOH
func.func @DepthConvAssignedSOH(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0, %cst) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 18 : i64, multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x14x14xf16, {order = #NHWC}>
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

// CHECK-LABEL: @MaxPoolAssignedSOH
func.func @MaxPoolAssignedSOH(%arg0: tensor<1x32x14x14xf16, {order = #NHWC}>) -> tensor<1x32x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], multiClusterStrategy = "SplitOverHeight", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedClustering
func.func @MaxPoolAssignedClustering(%arg0: tensor<1x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x3x3xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %0 = VPU.NCE.MaxPool(%arg0, %cst_0, %cst) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x3x3xf16, {order = #NHWC}>
    return %0 : tensor<1x32x3x3xf16, {order = #NHWC}>

    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<1x1x1x16xui8> = dense<10> : tensor<1x1x1x16xui8>
    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0, [[WEIGHTSTABLE]], [[ACTIVATION_WINDOW]])
    //CHECK-SAME:   {activation_window_channel_length = 4 : i64, kernel_size = [1, 1], multiClusterStrategy = "Clustering", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x3x3xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x3x3xf16, {order = #NHWC}>
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
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights>

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
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1024x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1024x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1024x1x1x128xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = "SplitOverKernel",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [1024, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1024x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                sparsity_map=tensor<1x1024x1x1xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvSubgraphOptimizationSplitOverKernel
func.func @ConvSubgraphOptimizationSplitOverKernel(%arg0: tensor<1x64x12x12xf16, {order = #NHWC}>) -> tensor<1x64x3x3xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    // supposed to be SOH without subgraph optimization
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 64, 2, 2], strides = [2, 2]} -> tensor<1x64x6x6xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst1) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 64, 2, 2], strides = [2, 2]} -> tensor<1x64x3x3xf16, {order = #NHWC}>
    return %1 : tensor<1x64x3x3xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.*]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverKernel"

    //CHECK:        [[WEIGHTSTABLE1:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.*]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverKernel"

    //CHECK:        return [[VAL1]] : tensor<1x64x3x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvSubgraphOptimizationClustering
func.func @ConvSubgraphOptimizationClustering(%arg0: tensor<1x80x22x22xf16, {order = #NHWC}>) -> tensor<1x48x22x22xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x80x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 80, 3, 3], strides = [1, 1]} -> tensor<1x80x22x22xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare tensor<64x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x80x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst1) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 80, 3, 3], strides = [1, 1]} -> tensor<1x64x22x22xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_2 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = VPU.NCE.Convolution(%1, %cst_2, %cst2) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x22x22xf16, {order = #NHWC}>

    return %2 : tensor<1x48x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.*]] = const.Declare tensor<80x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x80x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverHeight"

    //CHECK:        [[WEIGHTSTABLE1:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.*]] = const.Declare tensor<64x80x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x80x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverHeight"

    //CHECK:        [[WEIGHTSTABLE2:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS2:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverHeight"

    //CHECK:        return [[VAL2]] : tensor<1x48x22x22xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvSubgraphOptimizationSOK
func.func @SparseConvSubgraphOptimizationSOK(%arg0 : tensor<1x64x12x12xf16, {order = #NHWC}>, %arg1 : tensor<1x64x12x12xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x3x3xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x12x12xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x12x12xi1, {order = #NHWC}>>

    %weights1 = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights1_sm = const.Declare tensor<64x1x1x256xi1> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights1_sparse = VPU.GroupSparseTensor(%weights1, %weights1_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x2x2xf16, {order = #NHWC}>,
                             sparsity_map=tensor<64x1x1x256xi1>, is_weights>

    %weights_table1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights1_sparse, %weights_table1) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [64, 64, 2, 2],
            strides = [2, 2]
        } -> !VPU.SparseTensor<data=tensor<1x64x6x6xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x64x6x6xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 6, 6] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    %weights2 = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights2_sm = const.Declare tensor<64x1x1x256xi1> = dense<2.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights2_sparse = VPU.GroupSparseTensor(%weights2, %weights2_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x2x2xf16, {order = #NHWC}>,
                             sparsity_map=tensor<64x1x1x256xi1>, is_weights>

    %weights_table2 = const.Declare tensor<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>

    %1 = VPU.NCE.Convolution(%0, %weights2_sparse, %weights_table2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [64, 64, 2, 2],
            strides = [2, 2]
        } -> !VPU.SparseTensor<data=tensor<1x64x3x3xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x64x3x3xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 3, 3] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        }

    return %1 : !VPU.SparseTensor<data=tensor<1x64x3x3xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x64x3x3xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x12x12xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x12x12xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS1:%.+]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST_WEIGHTS1_SM:%.+]] = const.Declare tensor<64x1x1x256xi1>
    // CHECK:       [[WEIGHTS1_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS1]], [[CST_WEIGHTS1_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<64x64x2x2xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<64x1x1x256xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK:       [[OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS1_SPARSE]], [[CST_WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x6x6xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x6x6xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS2:%.+]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST_WEIGHTS2_SM:%.+]] = const.Declare tensor<64x1x1x256xi1>
    // CHECK:       [[WEIGHTS2_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS2]], [[CST_WEIGHTS2_SM]]) {is_weights}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<64x64x2x2xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<64x1x1x256xi1>, is_weights>

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE2:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>

    // CHECK:       [[OUT2:%.+]] = VPU.NCE.Convolution([[OUT1]], [[WEIGHTS2_SPARSE]], [[CST_WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x3x3xi1, {order = #NHWC}>>

    // CHECK:       return [[OUT2]] : !VPU.SparseTensor<data=tensor<1x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                                      sparsity_map=tensor<1x64x3x3xi1, {order = #NHWC}>>
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

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
