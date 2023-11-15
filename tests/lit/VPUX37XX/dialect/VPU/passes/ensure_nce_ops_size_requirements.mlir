//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeLargeHeight
module @PermuteQuantizeLargeHeight attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

func.func @TileByH(%arg0: tensor<1x32x8208x2xf16, {order = #NHWC}>) -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}> {
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
    } -> tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x32x8208x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 4104, 2] :
    // CHECK-SAME:      tensor<1x32x8208x2xf16, {order = #NHWC}> to tensor<1x32x4104x2xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x32x4104x2x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 4104, 0] [1, 32, 4104, 2] :
    // CHECK-SAME:      tensor<1x32x8208x2xf16, {order = #NHWC}> to tensor<1x32x4104x2xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantizeLargeChannel
module @PermuteQuantizeLargeChannel attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

func.func @TileByC(%arg0: tensor<1x16304x16x1xf16, {order = #NHWC}>) -> tensor<1x16304x16x1x!qElemType, {order = #NWCH}> {
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
    } -> tensor<1x16304x16x1x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x16304x16x1x!qElemType, {order = #NWCH}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 8160, 16, 1] :
    // CHECK-SAME:      tensor<1x16304x16x1xf16, {order = #NHWC}> to tensor<1x8160x16x1xf16, {order = #NHWC}>

    // CHECK:       [[FIRST_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x8160x16x1x!qElemType, {order = #NWCH}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 8160, 0, 0] [1, 8144, 16, 1] :
    // CHECK-SAME:      tensor<1x16304x16x1xf16, {order = #NHWC}> to tensor<1x8144x16x1xf16, {order = #NHWC}>

    // CHECK:       [[SECOND_QUANT_PERM:%.*]] = VPU.NCE.PermuteQuantize([[SECOND_SLICE]])
    // CHECK-SAME:      -> tensor<1x8144x16x1x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_QUANT_PERM]], [[SECOND_QUANT_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 8160, 0, 0]]
    // CHECK-SAME:  } :
    // CHECK-SAME:  tensor<1x8160x16x1x!qElemType, {order = #NWCH}>,
    // CHECK-SAME:  tensor<1x8144x16x1x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x16304x16x1x!qElemType, {order = #NWCH}>

    // CHECK:       [[CONCAT]] : tensor<1x16304x16x1x!qElemType, {order = #NWCH}>
}

}
