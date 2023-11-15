//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --propagate-expand %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8:f16, 0.010857077205882353:127>
!qElemType1 = !quant.uniform<u8:f16, 0.018435968137254902:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0082261029411764708:127>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {0.0038966381643700788:127,0.0046252153051181098:127,0.0039408526082677165:127,0.0037697619340551179:127,0.0032103531003937007:127,0.0037832185039370077:127,0.0035102423720472439:127,0.0035986712598425198:127,0.0036063607283464568:127,0.0038889486958661418:127,0.0055940883366141728:127,0.0032661017470472439:127,0.0032661017470472439:127,0.0032661017470472439:127,0.0032661017470472439:127,0.0032661017470472439:127}>


// CHECK-LABEL: @PropagateExpandFullCase
func.func @PropagateExpandFullCase(%arg0: tensor<1x3x512x512xf16>, %arg1: tensor<1x32x256x256x!qElemType0, {order = #NHWC}>) -> (tensor<1x16x128x128x!qElemType0, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}>) {
    %cst = const.Declare tensor<16x48x3x3x!qElemType3, {order = #NHWC}> = dense<1.0> : tensor<16x48x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType3>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<16x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType3>, #const.Reorder<#NHWC>]
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType0, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    %1 = IE.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> tensor<1x48x128x128x!qElemType0, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.30000001192092896 : f64}>, strides = [1, 1]} : tensor<1x48x128x128x!qElemType0, {order = #NHWC}>, tensor<16x48x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x16x128x128x!qElemType0, {order = #NHWC}>
    %3 = IE.Convolution(%arg1, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.30000001192092896 : f64}>, strides = [1, 1]} : tensor<1x32x256x256x!qElemType0, {order = #NHWC}>, tensor<16x32x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType0, {order = #NHWC}>
    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 12, 256, 256] : tensor<1x16x256x256x!qElemType0, {order = #NHWC}> to tensor<1x12x256x256x!qElemType0, {order = #NHWC}>
    %5 = IE.DepthToSpace(%4) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x12x256x256x!qElemType0, {order = #NHWC}> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    %6 = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs(%0 : tensor<1x3x512x512x!qElemType0, {order = #NHWC}>) -> tensor<1x16x256x192x!qElemType0, {order = #NHWC}>
    %7 = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs(%5 : tensor<1x3x512x512x!qElemType0, {order = #NHWC}>) -> tensor<1x16x256x192x!qElemType0, {order = #NHWC}>
    %8 = IE.Add(%6, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192x!qElemType0, {order = #NHWC}>, tensor<1x16x256x192x!qElemType0, {order = #NHWC}> -> tensor<1x16x256x192x!qElemType1, {order = #NHWC}>
    %9 = IE.ShapeCast {shape = [1, 3, 512, 512]} inputs(%8 : tensor<1x16x256x192x!qElemType1, {order = #NHWC}>) -> tensor<1x3x512x512x!qElemType1, {order = #NHWC}>
    %10 = IE.Expand(%9) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
    return %2, %10: tensor<1x16x128x128x!qElemType0, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}>

    //CHECK:    [[CST:%.*]] =   const.Declare tensor<64x4x4x4x!qElemType2, {order = #NHWC}>
    //CHECK-SAME:  [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    //CHECK:    [[CST_0:%.*]] = const.Declare tensor<16x64x3x3x!qElemType3, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x48x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType3>, #const.Reorder<#NHWC>, #const.Reshape<[16, 3, 48, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.Reshape<[16, 64, 3, 3]>]
    //CHECK:    [[CST_1:%.*]] = const.Declare tensor<1x16x256x256x!qElemType0, {order = #NHWC}> = dense<1.270000e+02> : tensor<1x16x256x256xf32>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>, #const.Reorder<#NHWC>]
    //CHECK:    [[CST_2:%.*]] = const.Declare tensor<16x16x2x2x!qElemType2, {order = #NHWC}>
    //CHECK-SAME: [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    //CHECK:    [[CST_3:%.*]] = const.Declare tensor<16x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType3>, #const.Reorder<#NHWC>]
    //CHECK:    [[PQ_0:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType0, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[EXPAND:%.*]] = IE.Expand([[PQ_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 256]} inputs([[EXPAND]] : tensor<1x4x512x512x!qElemType0, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType0, {order = #NHWC}>
    //CHECK:    [[S2D:%.*]] = IE.Convolution([[EXPAND]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [4, 4]} : tensor<1x4x512x512x!qElemType0, {order = #NHWC}>, tensor<64x4x4x4x!qElemType2, {order = #NHWC}> -> tensor<1x64x128x128x!qElemType0, {order = #NHWC}>
    //CHECK:    [[CONV_1:%.*]] = IE.Convolution([[S2D]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.30000001192092896 : f64}>, strides = [1, 1]} : tensor<1x64x128x128x!qElemType0, {order = #NHWC}>, tensor<16x64x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x16x128x128x!qElemType0, {order = #NHWC}>
    //CHECK:    [[CONV_2:%.*]] = IE.Convolution(%arg1, [[CST_3]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.30000001192092896 : f64}>, strides = [1, 1]} : tensor<1x32x256x256x!qElemType0, {order = #NHWC}>, tensor<16x32x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType0, {order = #NHWC}>
    //CHECK:    [[CONCAT:%.*]] = IE.Concat([[CONV_2]], [[CST_1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x256x256x!qElemType0, {order = #NHWC}>, tensor<1x16x256x256x!qElemType0, {order = #NHWC}> -> tensor<1x16x512x256x!qElemType0, {order = #NHWC}>
    //CHECK:    [[CONV_3:%.*]] = IE.Convolution([[CONCAT]], [[CST_2]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x16x512x256x!qElemType0, {order = #NHWC}>, tensor<16x16x2x2x!qElemType2, {order = #NHWC}> -> tensor<1x16x512x128x!qElemType0, {order = #NHWC}>
    //CHECK:    [[RESHAPE:%.*]] = IE.AffineReshape([[CONV_3]])
    //CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 4, 512, 512]} : tensor<1x16x512x128x!qElemType0, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_1:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 256]} inputs([[RESHAPE]] : tensor<1x4x512x512x!qElemType0, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType0, {order = #NHWC}>
    //CHECK:    [[ADD:%.*]] =  IE.Add([[SHAPECAST_0]], [[SHAPECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256x!qElemType0, {order = #NHWC}>, tensor<1x16x256x256x!qElemType0, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType1, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_2:%.*]] = IE.ShapeCast {shape = [1, 4, 512, 512]} inputs([[ADD]] : tensor<1x16x256x256x!qElemType1, {order = #NHWC}>) -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
    //CHECK:    return [[CONV_1]], [[SHAPECAST_2]] : tensor<1x16x128x128x!qElemType0, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8:f16, 0.018435968137254902:128>
!qElemType1 = !quant.uniform<u8:f16, 0.010857077205882353:127>


// CHECK-LABEL: @PropagateExpandPermuteQuantize
func.func @PropagateExpandPermuteQuantize(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x4x512x512x!qElemType0, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType1, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType1, {order = #NHWC}>
    %1 = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs(%0 : tensor<1x3x512x512x!qElemType1, {order = #NHWC}>) -> tensor<1x16x256x192x!qElemType1, {order = #NHWC}>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192x!qElemType1, {order = #NHWC}>, tensor<1x16x256x192x!qElemType1, {order = #NHWC}> -> tensor<1x16x256x192x!qElemType0, {order = #NHWC}>
    %3 = IE.ShapeCast {shape = [1, 3, 512, 512]} inputs(%2 : tensor<1x16x256x192x!qElemType0, {order = #NHWC}>) -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
    return %4: tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
        
    
    //CHECK:    [[PQ_0:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType1, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType1, {order = #NHWC}>
    //CHECK:    [[EXPAND:%.*]] = IE.Expand([[PQ_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 256]} inputs([[EXPAND]] : tensor<1x4x512x512x!qElemType1, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType1, {order = #NHWC}>
    //CHECK:    [[ADD:%.*]] = IE.Add([[SHAPECAST_0]], [[SHAPECAST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256x!qElemType1, {order = #NHWC}>, tensor<1x16x256x256x!qElemType1, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType0, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_1:%.*]] = IE.ShapeCast {shape = [1, 4, 512, 512]} inputs([[ADD]] : tensor<1x16x256x256x!qElemType0, {order = #NHWC}>) -> tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    return [[SHAPECAST_1]] : tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8:f16, 0.010857077205882353:127>
!qElemType1 = !quant.uniform<u8:f16, 0.018435968137254902:128>

// CHECK-LABEL: @PropagateExpandPermuteQuantizeSlice
func.func @PropagateExpandPermuteQuantizeSlice(%arg0: tensor<1x3x512x512xf16>) -> (tensor<1x3x512x512x!qElemType0, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}>) {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType0, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    %1 = IE.MaxPool(%0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], rounding_type = #IE.rounding_type<FLOOR>} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    %2 = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs(%0 : tensor<1x3x512x512x!qElemType0, {order = #NHWC}>) -> tensor<1x16x256x192x!qElemType0, {order = #NHWC}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x192x!qElemType0, {order = #NHWC}>, tensor<1x16x256x192x!qElemType0, {order = #NHWC}> -> tensor<1x16x256x192x!qElemType1, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 3, 512, 512]} inputs(%3 : tensor<1x16x256x192x!qElemType1, {order = #NHWC}>) -> tensor<1x3x512x512x!qElemType1, {order = #NHWC}>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
    return %1, %5: tensor<1x3x512x512x!qElemType0, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
        
    
    //CHECK:    [[PQ_0:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType0, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[EXPAND:%.*]] = IE.Expand([[PQ_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[SLICE:%.*]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 3, 512, 512] : tensor<1x4x512x512x!qElemType0, {order = #NHWC}> to tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_0:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 256]} inputs([[EXPAND]] : tensor<1x4x512x512x!qElemType0, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType0, {order = #NHWC}>
    //CHECK:    [[MAXPOOL:%.*]] = IE.MaxPool([[SLICE]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> tensor<1x3x512x512x!qElemType0, {order = #NHWC}>
    //CHECK:    [[ADD:%.*]] = IE.Add([[SHAPECAST_0]], [[SHAPECAST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256x!qElemType0, {order = #NHWC}>, tensor<1x16x256x256x!qElemType0, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType1, {order = #NHWC}>
    //CHECK:    [[SHAPECAST_1:%.*]] = IE.ShapeCast {shape = [1, 4, 512, 512]} inputs([[ADD]] : tensor<1x16x256x256x!qElemType1, {order = #NHWC}>) -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
    //CHECK:    return [[MAXPOOL]], [[SHAPECAST_1]] : tensor<1x3x512x512x!qElemType0, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
}
