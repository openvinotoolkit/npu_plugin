//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @SplitNCEConvOverOW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x1x19627xf16, {order = #NHWC}>
func @SplitNCEConvOverOW(%input: tensor<1x16x1x19627xf16, {order = #NHWC}>)
                        -> tensor<1x16x1x19627xf16, {order = #NHWC}> {
    %weightsTable = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %filter = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.200000e+01> : tensor<1xf16>,
        [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>,
        #const.Reorder<#NCHW>, #const.Reshape<[16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %activationWindow = const.Declare tensor<1x1x1x16xui8> = 
        dense<[[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>
   
    %1 = VPU.NCE.DepthConvolution(%input, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 4 : i64, multiClusterStrategy = "Clustering",
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, mode = "NOOP"},
        rawFilterShape = [16, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 9]}
        -> tensor<1x16x1x19627xf16, {order = #NHWC}>

    return %1 : tensor<1x16x1x19627xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.200000e+01>
    // CHECK-SAME:      : tensor<1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>,
    // CHECK-SAME:      #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>, #const.Reorder<#NCHW>,
    // CHECK-SAME:      #const.Reshape<[16, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 1, 6543]
    // CHECK-SAME:      : tensor<1x16x1x19627xf16, {order = #NHWC}> to tensor<1x16x1x6543xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.DepthConvolution([[ACTIVATION_TILE_0]], [[FILTER]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          {activation_window_channel_length = 4 : i64, multiClusterStrategy = "Clustering",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:                 fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:          rawFilterShape = [16, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:          -> tensor<1x16x1x6543xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 6543] [1, 16, 1, 6542]
    // CHECK-SAME:      : tensor<1x16x1x19627xf16, {order = #NHWC}> to tensor<1x16x1x6542xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.DepthConvolution([[ACTIVATION_TILE_1]], [[FILTER]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          {activation_window_channel_length = 4 : i64, multiClusterStrategy = "Clustering",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:                 fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:          rawFilterShape = [16, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:          -> tensor<1x16x1x6542xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 13085] [1, 16, 1, 6542]
    // CHECK-SAME:      : tensor<1x16x1x19627xf16, {order = #NHWC}> to tensor<1x16x1x6542xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE2:%.+]] = VPU.NCE.DepthConvolution([[ACTIVATION_TILE_2]], [[FILTER]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          {activation_window_channel_length = 4 : i64, multiClusterStrategy = "Clustering",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:                 fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:          rawFilterShape = [16, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:          -> tensor<1x16x1x6542xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 6543], [0, 0, 0, 13085]
    // CHECK-SAME:          -> tensor<1x16x1x19627xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x19627xf16, {order = #NHWC}>
}
