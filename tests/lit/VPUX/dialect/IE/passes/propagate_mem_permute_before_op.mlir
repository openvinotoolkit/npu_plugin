//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-before-op --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @PropagateMemPermuteThroughAffineReshape(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x1280x4096x1xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    return %1 : tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1280x1x4096xf16> -> tensor<1x4096x1280x1xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 4096, 1, 1280]} : tensor<1x4096x1280x1xf16> -> tensor<1x4096x1x1280xf16>
    // CHECK:               [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTECAST]] : tensor<1x1280x4096x1xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @PropagateMemPermuteThroughAffineReshapeWithPermNWHC(%arg0: tensor<1x64x64x320xf16, {order = #NWCH}>) -> tensor<1x4096x320x1xf16, {order = #NCWH}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 320, 1]} : tensor<1x64x64x320xf16, {order = #NWCH}> -> tensor<1x4096x320x1xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x4096x320x1xf16, {order = #NHWC}> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    return %1 : tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:                [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x64x64x320xf16, {order = #NWCH}> -> tensor<1x64x64x320xf16>
    // CHECK:                [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 1, 320]} : tensor<1x64x64x320xf16> -> tensor<1x4096x1x320xf16>
    // CHECK:                [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x4096x1x320xf16> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:                return [[PERMUTECAST]] : tensor<1x4096x320x1xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @NotPropagateMemPermuteIfBrokenSplitAxis(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x1280x2048x2xf16, {order = #NHCW}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 2048, 2]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x2048x2xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHCW, mem_perm = #NHCW} : tensor<1x1280x2048x2xf16> -> tensor<1x1280x2048x2xf16, {order = #NHCW}>

    return %1 : tensor<1x1280x2048x2xf16, {order = #NHCW}>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 2048, 2]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x2048x2xf16>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute([[RESHAPE]]) {dst_order = #NHCW, mem_perm = #NHCW} : tensor<1x1280x2048x2xf16> -> tensor<1x1280x2048x2xf16, {order = #NHCW}>

    // CHECK:               return [[MEMPERMUTE]] : tensor<1x1280x2048x2xf16, {order = #NHCW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @PropagateMemPermuteIfBreakSplitAxisWithSingleNonTrivialMemShape(%arg0: tensor<1x8x4096x40xf16>) -> tensor<8x40x4096x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [8, 4096, 40, 1]} : tensor<1x8x4096x40xf16> -> tensor<8x4096x40x1xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<8x4096x40x1xf16> -> tensor<8x40x4096x1xf16>

    return %1 : tensor<8x40x4096x1xf16>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x4096x40xf16> -> tensor<1x8x40x4096xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [8, 40, 4096, 1]} : tensor<1x8x40x4096xf16> -> tensor<8x40x4096x1xf16>

    // CHECK:               return [[RESHAPE]] : tensor<8x40x4096x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @PropagateMemPermuteInCaseSplitAxisIsTrivial(%arg0: tensor<1x768x14x14xf16, {order = #NHWC}>) -> tensor<1x1x196x768xf16, {order = #NCWH}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 768, 196]} : tensor<1x768x14x14xf16, {order = #NHWC}> -> tensor<1x1x768x196xf16, {order = #NCWH}>
    %1 = IE.MemPermute(%0) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x768x196xf16, {order = #NCWH}> -> tensor<1x1x196x768xf16, {order = #NCWH}>

    return %1 : tensor<1x1x196x768xf16, {order = #NCWH}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x768x14x14xf16, {order = #NHWC}> -> tensor<1x768x14x14xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 768, 196]} : tensor<1x768x14x14xf16> -> tensor<1x1x768x196xf16>
    // CHECK:               [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x1x768x196xf16> -> tensor<1x1x196x768xf16, {order = #NCWH}>

    // CHECK:               return [[PERMUTECAST]] : tensor<1x1x196x768xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @PropagateMemPermuteThroughAffineReshapeNHWCInput(%arg0: tensor<1x4096x1x320xf16, {order = #NHWC}>) -> tensor<1x64x64x320xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 64, 64, 320]} : tensor<1x4096x1x320xf16, {order = #NHWC}> -> tensor<1x64x64x320xf16, {order = #NWCH}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x64x64x320xf16, {order = #NWCH}> -> tensor<1x64x64x320xf16>

    return %1 : tensor<1x64x64x320xf16>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4096x1x320xf16, {order = #NHWC}> -> tensor<1x4096x1x320xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 64, 64, 320]} : tensor<1x4096x1x320xf16> -> tensor<1x64x64x320xf16>

    // CHECK:               return [[RESHAPE]] : tensor<1x64x64x320xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @PropagatePermuteQuantizeThroughAffineReshape(%arg0: tensor<1x4096x1x1280xf16>) -> tensor<1x1280x4096x1xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x1x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[IN_PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x4096x1x1280xf16> -> tensor<1x4096x1280x1xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[IN_PERMUTECAST]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 4096, 1, 1280]} : tensor<1x4096x1280x1xf16> -> tensor<1x4096x1x1280xf16>
    // CHECK:               [[OUT_PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               return [[OUT_PERMUTECAST]] : tensor<1x1280x4096x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @NotPropagatePermuteQuantizeIfNotBeneficial(%arg0: tensor<1x4096x4x1280xf16>) -> tensor<1280x1x4096x4xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x4x1280xf16> -> tensor<1x1280x4x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1280, 1, 4, 4096]} : tensor<1x1280x4x4096xf16> -> tensor<1280x1x4x4096xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NCHW, mem_perm = #NCWH, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1280x1x4x4096xf16> -> tensor<1280x1x4096x4xf16>

    return %2 : tensor<1280x1x4096x4xf16>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x4x1280xf16> -> tensor<1x1280x4x4096xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1280, 1, 4, 4096]} : tensor<1x1280x4x4096xf16> -> tensor<1280x1x4x4096xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = f16, dst_order = #NCHW, mem_perm = #NCWH, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1280x1x4x4096xf16> -> tensor<1280x1x4096x4xf16>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1280x1x4096x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NotPropagatePermuteQuantizeIfChangesElemType(%arg0: tensor<1x4096x1x1280xf16>) -> tensor<1x1280x4096x1x!qElemType, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x1x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>

    return %2 : tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x1x4096xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotPropagatePermuteQuantizeForInvalidRank(%arg0: tensor<4096x1x1280xf16>) -> tensor<1x1280x4096x1xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #CHW, mem_perm = #map} : tensor<4096x1x1280xf16> -> tensor<1280x4096x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1280, 4096, 1]} : tensor<1280x4096x1xf16> -> tensor<1x1280x4096x1xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #CHW, mem_perm = #map} : tensor<4096x1x1280xf16> -> tensor<1280x4096x1xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1280, 4096, 1]} : tensor<1280x4096x1xf16> -> tensor<1x1280x4096x1xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1x1280x4096x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @MoveThroughMVN(%arg0: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x4096x1xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #map, mem_perm = #map} : tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x320xf16, {order = #map}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 320, 1]} : tensor<1x64x64x320xf16, {order = #map}> -> tensor<1x4096x320x1xf16, {order = #NHWC}>
    %2 = IE.MVN(%1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x4096x320x1xf16, {order = #NHWC}> -> tensor<1x4096x320x1xf16, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x4096x320x1xf16, {order = #NHWC}> -> tensor<1x320x4096x1xf16, {order = #NHWC}>
    return %3 : tensor<1x320x4096x1xf16, {order = #NHWC}>


    // CHECK:               [[PERMUTECAST0:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x320xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[PERMUTECAST0]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 1, 320]} : tensor<1x64x64x320xf16> -> tensor<1x4096x1x320xf16>
    // CHECK:               [[PERMUTECAST1:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x4096x1x320xf16> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:               [[MVN:%.*]] = IE.MVN([[PERMUTECAST1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x4096x320x1xf16, {order = #NCWH}> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:               [[PERMUTECAST2:%.*]] = IE.PermuteCast([[MVN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x320x1xf16, {order = #NCWH}> -> tensor<1x320x4096x1xf16, {order = #NHWC}>
    // CHECK:               return [[PERMUTECAST2]] : tensor<1x320x4096x1xf16, {order = #NHWC}>
}
