//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --prefetch-tiling --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s

// This operation needs separate lit-test file because vpux::VPU::verifyOp(NCEPermuteQuantizeOp) fails to check arch.
// See E#60343 for details. Merge this file into prefetch_tiling.mlir when the ticket is resolved.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantize
module @PermuteQuantize attributes {VPU.arch = "VPUX37XX"} {
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

func @SplitPermuteQuantize(%arg0: tensor<1x32x3x8208xf16, {order = #NHWC}>) -> tensor<1x32x4x8208x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = {
            bottom = 1 : i64,
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64
        },
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "NOOP"
        }
    } -> tensor<1x32x4x8208x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x4x8208x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 3, 4104] :
    // CHECK-SAME:      tensor<1x32x3x8208xf16, {order = #NHWC}> to tensor<1x32x3x4104xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      -> tensor<1x32x4x4104x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 4104] [1, 32, 3, 4104] :
    // CHECK-SAME:      tensor<1x32x3x8208xf16, {order = #NHWC}> to tensor<1x32x3x4104xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      -> tensor<1x32x4x4104x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_QUANT_PERM]], [[SECOND_QUANT_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4104]]
    // CHECK-SAME:  } :
    // CHECK-SAME:  tensor<1x32x4x4104x!qElemType, {order = #NWCH}>,
    // CHECK-SAME:  tensor<1x32x4x4104x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x32x4x8208x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT]] : tensor<1x32x4x8208x!qElemType, {order = #NWCH}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeDoesNotFitCMX
module @PermuteQuantizeDoesNotFitCMX attributes {VPU.arch = "VPUX37XX"} {
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

func @SplitPermuteQuantize(%arg0: tensor<1x32x16x2048xf16, {order = #NHWC}>) -> tensor<1x32x16x2048x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = {
            bottom = 0 : i64,
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64
        },
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "NOOP"
        }
    } -> tensor<1x32x16x2048x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x16x2048x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 16, 1024] :
    // CHECK-SAME:      tensor<1x32x16x2048xf16, {order = #NHWC}> to tensor<1x32x16x1024xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      -> tensor<1x32x16x1024x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 1024] [1, 32, 16, 1024] :
    // CHECK-SAME:      tensor<1x32x16x2048xf16, {order = #NHWC}> to tensor<1x32x16x1024xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      -> tensor<1x32x16x1024x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_QUANT_PERM]], [[SECOND_QUANT_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1024]]
    // CHECK-SAME:  } :
    // CHECK-SAME:  tensor<1x32x16x1024x!qElemType, {order = #NWCH}>,
    // CHECK-SAME:  tensor<1x32x16x1024x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x32x16x2048x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT]] : tensor<1x32x16x2048x!qElemType, {order = #NWCH}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeLargeHeight
module @PermuteQuantizeLargeHeight attributes {VPU.arch = "VPUX37XX"} {
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}

func @TileByH(%arg0: tensor<1x32x8208x2xf16, {order = #NHWC}>) -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = {
            bottom = 0 : i64,
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64
        },
        ppe = {
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "NOOP"
        }
    } -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 4104, 2] :
    // CHECK-SAME:      tensor<1x32x8208x2xf16, {order = #NHWC}> to tensor<1x32x4104x2xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x32x4104x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 4104, 0] [1, 32, 4104, 2] :
    // CHECK-SAME:      tensor<1x32x8208x2xf16, {order = #NHWC}> to tensor<1x32x4104x2xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x32x4104x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_QUANT_PERM]], [[SECOND_QUANT_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 4104, 0]]
    // CHECK-SAME:  } :
    // CHECK-SAME:  tensor<1x32x4104x2x!qElemType, {order = #NWCH}>,
    // CHECK-SAME:  tensor<1x32x4104x2x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT]] : tensor<1x32x8208x2x!qElemType, {order = #NWCH}>
}

}
