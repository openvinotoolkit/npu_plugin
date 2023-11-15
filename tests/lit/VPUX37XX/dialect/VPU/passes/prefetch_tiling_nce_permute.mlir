//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --tiling="enable-prefetch=true" --canonicalize %s | FileCheck %s

// This operation needs separate lit-test file because vpux::VPU::NCEPermuteQuantizeOp::verify() fails to check arch.
// See E#60343 for details. Merge this file into prefetch_tiling.mlir when the ticket is resolved.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantize
module @PermuteQuantize attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

func.func @SplitPermuteQuantize(%arg0: tensor<1x32x3x8208xf16, {order = #NHWC}>) -> tensor<1x32x4x8208x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x32x4x8208x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x4x8208x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 3, 4104] :
    // CHECK-SAME:      tensor<1x32x3x8208xf16, {order = #NHWC}> to tensor<1x32x3x4104xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x32x4x4104x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 4104] [1, 32, 3, 4104] :
    // CHECK-SAME:      tensor<1x32x3x8208xf16, {order = #NHWC}> to tensor<1x32x3x4104xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
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

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeDoesNotFitCMX
module @PermuteQuantizeDoesNotFitCMX attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

func.func @SplitPermuteQuantize(%arg0: tensor<1x32x16x2048xf16, {order = #NHWC}>) -> tensor<1x32x16x2048x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x32x16x2048x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x16x2048x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 16, 1024] :
    // CHECK-SAME:      tensor<1x32x16x2048xf16, {order = #NHWC}> to tensor<1x32x16x1024xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x32x16x1024x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 1024] [1, 32, 16, 1024] :
    // CHECK-SAME:      tensor<1x32x16x2048xf16, {order = #NHWC}> to tensor<1x32x16x1024xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
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

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeLargeHeight
module @PermuteQuantizeLargeHeight attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

func.func @TileByH(%arg0: tensor<1x32x8208x4xf16, {order = #NHWC}>) -> tensor<1x32x8208x4x!qElemType, {order = #NWCH}> {
    %0 = VPU.NCE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NWCH,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPETask<
            clamp_high = 255 : i64,
            clamp_low = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = <NOOP>
        >
    } -> tensor<1x32x8208x4x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x8208x4x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 8208, 2] :
    // CHECK-SAME:      tensor<1x32x8208x4xf16, {order = #NHWC}> to tensor<1x32x8208x2xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 2] [1, 32, 8208, 2] :
    // CHECK-SAME:      tensor<1x32x8208x4xf16, {order = #NHWC}> to tensor<1x32x8208x2xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
    // CHECK-SAME:      -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_QUANT_PERM]], [[SECOND_QUANT_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 2]]
    // CHECK-SAME:  } :
    // CHECK-SAME:  tensor<1x32x8208x2x!qElemType, {order = #NWCH}>,
    // CHECK-SAME:  tensor<1x32x8208x2x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x32x8208x4x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT]] : tensor<1x32x8208x4x!qElemType, {order = #NWCH}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @NCEPermute
module @NCEPermute attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

func.func @SplitNCEPermute(%arg0: tensor<1x3x32x8000xf16>) -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64
    } -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x32x8000x!qElemType, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 3, 16, 8000] 
    // CHECK-SAME:      : tensor<1x3x32x8000xf16> to tensor<1x3x16x8000xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Permute([[INPUT_TILE0]]) 
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 4 : i64}
    // CHECK-SAME:      -> tensor<1x4x16x8000x!qElemType, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 16, 0] [1, 3, 16, 8000]
    // CHECK-SAME:      : tensor<1x3x32x8000xf16> to tensor<1x3x16x8000xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Permute([[INPUT_TILE1]]) 
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 4 : i64}
    // CHECK-SAME:      -> tensor<1x4x16x8000x!qElemType, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 16, 0]
    // CHECK-SAME:      -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x4x32x8000x!qElemType, {order = #NHWC}>
}

}
