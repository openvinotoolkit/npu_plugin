//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" --optimize-reorders %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSwMultiplyHasChange 
module @ReorderWithSwMultiplyHasChange {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
func.func @main(%arg0: tensor<1x32x28x1xf16, {order = #NHWC}>) -> tensor<1x32x28x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16>, tensor<1x32x28x1xf16> -> tensor<1x32x28x1xf16>

    return %2 : tensor<1x32x28x1xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x28x1xf16, {order = #NHWC}>, tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NCHW} : tensor<1x32x28x1xf16, {order = #NHWC}> -> tensor<1x32x28x1xf16>
    // CHECK:       return [[VAR1]] : tensor<1x32x28x1xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ReorderInNHWCPowerOp
module @ReorderInNHWCPowerOp {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x4x5x10xf16>) -> tensor<1x4x5x10xf16> {
func.func @main(%arg0: tensor<1x4x5x10xf16>) -> tensor<1x4x5x10xf16> {
    %cst = const.Declare tensor<1x10x1x1xf16> = dense<[[[[9.997550e-02]], [[3.000490e-01]], [[5.000000e-01]], [[7.001950e-01]], [[8.999020e-01]], [[1.099610e+00]], [[1.299800e+00]], [[1.500000e+00]], [[1.700200e+00]], [[1.900390e+00]]]]> : tensor<1x10x1x1xf16>
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x5x10xf16> -> tensor<1x10x4x5xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x10x4x5xf16, {order = #NHWC}> -> tensor<1x10x4x5xf16>
    %2 = IE.Power(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x4x5xf16>, tensor<1x10x1x1xf16> -> tensor<1x10x4x5xf16>
    %3 = IE.PermuteCast(%2) {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x10x4x5xf16> -> tensor<1x4x5x10xf16, {order = #NWCH}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<1x4x5x10xf16, {order = #NWCH}> -> tensor<1x4x5x10xf16>
    return %4 : tensor<1x4x5x10xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x10x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      tensor<1x10x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:           [[VAR0:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x5x10xf16> -> tensor<1x10x4x5xf16, {order = #NHWC}>
    // CHECK:           [[VAR1:%.+]] = IE.Power([[VAR0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x4x5xf16, {order = #NHWC}>, tensor<1x10x1x1xf16, {order = #NHWC}> -> tensor<1x10x4x5xf16, {order = #NHWC}>
    // CHECK:           [[VAR2:%.+]] = IE.PermuteCast([[VAR1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x10x4x5xf16, {order = #NHWC}> -> tensor<1x4x5x10xf16>
    // CHECK:           return [[VAR2]] : tensor<1x4x5x10xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNCHWLayoutForEfficiency 
module @MVNInNCHWLayoutForEfficiency {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x64x64x192xf16> {
    %cst = const.Declare tensor<64x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<64x4x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16>
    return %4 : tensor<1x64x64x192xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK:      [[REORDER:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
    // CHECK:      [[MVN:%.*]] = IE.MVN([[REORDER]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNHWCLayoutForEfficiency 
module @MVNInNHWCLayoutForEfficiency {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x384x16x48xf16> {
    %cst = const.Declare tensor<384x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [8, 8]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<384x4x3x3xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x384x16x48xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x384x16x48xf16> -> tensor<1x384x16x48xf16>
    return %4 : tensor<1x384x16x48xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x384x16x48xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16, {order = #NHWC}>
    // CHECK:      [[REORDER:%.*]] = IE.Reorder([[MVN]]) {dstOrder = #NCHW} : tensor<1x384x16x48xf16, {order = #NHWC}> -> tensor<1x384x16x48xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNHWCLayoutForAcrossChannels 
module @MVNInNHWCLayoutForAcrossChannels {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x64x64x192xf16> {
    %cst = const.Declare tensor<64x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<64x4x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
    %4 = IE.MVN(%3) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16>
    return %4 : tensor<1x64x64x192xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]]) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
    // CHECK:      [[REORDER:%.*]] = IE.Reorder([[MVN]]) {dstOrder = #NCHW} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVNInNHWCLayoutForWNotAligned 
module @MVNInNHWCLayoutForWNotAligned {
func.func @main(%arg0: tensor<1x4x128x384xf16, {order = #NHWC}>) -> tensor<1x64x64x191xf16> {
    %cst = const.Declare tensor<64x4x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x3x3xf16>, [#const.Reorder<#NHWC>]
    %2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x4x128x384xf16, {order = #NHWC}>, tensor<64x4x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x191xf16, {order = #NHWC}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x191xf16, {order = #NHWC}> -> tensor<1x64x64x191xf16>
    %4 = IE.MVN(%3) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x191xf16> -> tensor<1x64x64x191xf16>
    return %4 : tensor<1x64x64x191xf16>

    // CHECK:      [[CONV:%.*]] = IE.Convolution
    // CHECK:      [[REORDER:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x64x64x191xf16, {order = #NHWC}> -> tensor<1x64x64x191xf16>
    // CHECK:      [[MVN:%.*]] = IE.MVN([[REORDER]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x191xf16> -> tensor<1x64x64x191xf16>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @ReorderWithConcatNoSwap
module @ReorderWithConcatNoSwap {

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x768x64x64xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x768x64x64xf16, {order = #NHWC}>) ->  tensor<1x64x70x768xf16> {
    %cst = const.Declare tensor<768x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[768, 1, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1x768x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>]
    %cst_1 = const.Declare tensor<1x64x6x768xf16> = dense<0.000000e+00> : tensor<1x64x6x768xf16>
    %0 = IE.GroupConvolution(%arg0, %cst, %cst_0) {dilations = [1, 1], groups = 768 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<768x1x1x1xf16, {order = #NHWC}>, tensor<1x768x1x1xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #NWCH, mem_perm = #NCHW} : tensor<1x768x64x64xf16> -> tensor<1x64x64x768xf16, {order = #NWCH}>
    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x64x64x768xf16, {order = #NWCH}> -> tensor<1x64x64x768xf16>
    %4 = IE.Concat(%3, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]} : tensor<1x64x64x768xf16>, tensor<1x64x6x768xf16> -> tensor<1x64x70x768xf16>
    return %4 : tensor<1x64x70x768xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-SAME:      tensor<768x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>, #const.Reshape<[768, 1, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x768x1x1xf16> = dense<1.250000e-01> : tensor<1x1x1x768xf16>, [#const.Transpose<#NWCH>]

    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare
    // CHECK-SAME:      tensor<1x64x6x768xf16> = dense<0.000000e+00> : tensor<1x64x6x768xf16>

    // CHECK:       [[GROUP_CONV:%.*]] = IE.GroupConvolution([[ARG0]], [[CST]], [[CST_0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 768 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<768x1x1x1xf16, {order = #NHWC}>, tensor<1x768x1x1xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK-NOT:       IE.Reorder
    // CHECK:       [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[GROUP_CONV]])
    // CHECK-SAME:      {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x768xf16>
    // CHECK-NOT:       IE.Reorder

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[PERMUTE_CAST]], [[CST_1]])
    // CHECK-SAME:      tensor<1x64x64x768xf16>, tensor<1x64x6x768xf16> -> tensor<1x64x70x768xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x64x70x768xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwitchReorderWithTileForBetterTileDMAPerfCase2
func.func @SwitchReorderWithTileForBetterTileDMAPerfCase2(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x1x1xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    return %1 : tensor<1x64x11x11xf16>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[TILE]]) {dstOrder = #NCHW} : tensor<1x64x11x11xf16, {order = #NHWC}> -> tensor<1x64x11x11xf16>
    // CHECK:       return [[REORDER]] : tensor<1x64x11x11xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @FusingNonTrivialReorderAroundTile
func.func @FusingNonTrivialReorderAroundTile(%arg0: tensor<1x1x11x11xf16, {order = #NCWH}> ) -> tensor<1x64x11x11xf16, {order = #NCWH}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x11x11xf16, {order = #NCWH}> -> tensor<1x1x11x11xf16>
    %1 = IE.Tile(%0) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16> -> tensor<1x64x11x11xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11xf16, {order = #NCWH}>
    return %2 : tensor<1x64x11x11xf16, {order = #NCWH}>

    // CHECK:       [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 64, 1, 1]} : tensor<1x1x11x11xf16, {order = #NCWH}> -> tensor<1x64x11x11xf16, {order = #NCWH}>
    // CHECK        return [[TILE]] : tensor<1x64x11x11xf16, {order = #NCWH}>
}
