//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --prefetch-tiling --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16:3, {0.1:127, 0.2:127, 0.3:127, 0.4:127, 0.5:127, 0.6:127, 0.7:127, 0.8:127}>

// // 1x16x4x8xf16 + weights_table + act_window + profiling buffer
IE.MemoryResource 4800 bytes of @CMX_NN

// CHECK-LABEL: func @MultiAxesAndPerAxisQuant
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x8x8x!qElemType, {order = #NHWC}>
func @MultiAxesAndPerAxisQuant(
        %input: tensor<1x32x8x8x!qElemType, {order = #NHWC}>)
            -> tensor<1x32x8x8x!qElemType, {order = #NHWC}> {
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%input, %weights_table, %activation_window) {
        activation_window_channel_length = 54 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-SAME:      = dense<1> : tensor<32x1x1x4xsi32>

    // Tile 0, 1

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 3, 8]
    // CHECK-SAME:      : tensor<1x32x8x8x!qElemType, {order = #NHWC}> to tensor<1x32x3x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          activation_window_channel_length = 54 : i64,
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
    // CHECK-SAME:      -> tensor<1x32x2x8x!qElemType, {order = #NHWC}>

    // Tile 0, 2

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 1, 0] [1, 32, 4, 8]
    // CHECK-SAME:      : tensor<1x32x8x8x!qElemType, {order = #NHWC}> to tensor<1x32x4x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          activation_window_channel_length = 54 : i64,
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
    // CHECK-SAME:      -> tensor<1x32x2x8x!qElemType, {order = #NHWC}>

    // Tile 0, 3

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 3, 0] [1, 32, 4, 8]
    // CHECK-SAME:      : tensor<1x32x8x8x!qElemType, {order = #NHWC}> to tensor<1x32x4x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          activation_window_channel_length = 54 : i64,
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
    // CHECK-SAME:      -> tensor<1x32x2x8x!qElemType, {order = #NHWC}>

    // Tile 0, 4

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 5, 0] [1, 32, 3, 8]
    // CHECK-SAME:      : tensor<1x32x8x8x!qElemType, {order = #NHWC}> to tensor<1x32x3x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:          activation_window_channel_length = 54 : i64,
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
    // CHECK-SAME:      -> tensor<1x32x2x8x!qElemType, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]
    // CHECK-SAME:      -> tensor<1x32x8x8x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x8x8x!qElemType, {order = #NHWC}>
}
