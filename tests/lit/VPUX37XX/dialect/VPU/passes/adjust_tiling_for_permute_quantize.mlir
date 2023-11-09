//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --split-input-file --adjust-tiling-for-permute-quantize --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: module @PermuteQuantize
module @PermuteQuantize attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>} {
    IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    func.func @main(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x16x640x640x!qElemType, {order = #NHWC}> {
        %0 = VPU.Reshape(%arg0) {
            shape_value = [1, 640, 3, 640]
        } : tensor<1x3x640x640xf16> -> tensor<1x640x3x640xf16>

        %1 = VPU.LayoutCast(%0) {
            dst_order = #NHWC
        } : tensor<1x640x3x640xf16> -> tensor<1x640x3x640xf16, {order = #NHWC}>

        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 640, 3, 214] :
            tensor<1x640x3x640xf16, {order = #NHWC}> to tensor<1x640x3x214xf16, {order = #NHWC}>

        %3 = VPU.NCE.PermuteQuantize(%2) {
            dstElemType = !qElemType,
            dstOrder = #NWCH,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
            ppe = #VPU.PPETask<
                clamp_high = 255 : i64,
                clamp_low = 0 : i64,
                fp_prelu_alpha = 1.000000e+00 : f64,
                lrelu_mult = 1 : i64,
                lrelu_shift = 0 : i64,
                mode = <NOOP>
            >,
            tilingStrategy = [1, 1, 1, 3]
        } -> tensor<1x640x16x214x!qElemType, {order = #NWCH}>

        %4 = VPU.Slice %1 [0, 0, 0, 214] [1, 640, 3, 213] :
            tensor<1x640x3x640xf16, {order = #NHWC}> to tensor<1x640x3x213xf16, {order = #NHWC}>

        %5 = VPU.NCE.PermuteQuantize(%4) {
            dstElemType = !qElemType,
            dstOrder = #NWCH,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
            ppe = #VPU.PPETask<
                clamp_high = 255 : i64,
                clamp_low = 0 : i64,
                fp_prelu_alpha = 1.000000e+00 : f64,
                lrelu_mult = 1 : i64,
                lrelu_shift = 0 : i64,
                mode = <NOOP>
            >,
            tilingStrategy = [1, 1, 1, 3]
        } -> tensor<1x640x16x213x!qElemType, {order = #NWCH}>

        %6 = VPU.Slice %1 [0, 0, 0, 427] [1, 640, 3, 213] :
            tensor<1x640x3x640xf16, {order = #NHWC}> to tensor<1x640x3x213xf16, {order = #NHWC}>

        %7 = VPU.NCE.PermuteQuantize(%6) {
            dstElemType = !qElemType,
            dstOrder = #NWCH,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
            ppe = #VPU.PPETask<
                clamp_high = 255 : i64,
                clamp_low = 0 : i64,
                fp_prelu_alpha = 1.000000e+00 : f64,
                lrelu_mult = 1 : i64,
                lrelu_shift = 0 : i64,
                mode = <NOOP>
            >,
            tilingStrategy = [1, 1, 1, 3]
        } -> tensor<1x640x16x213x!qElemType, {order = #NWCH}>

        %8 = VPU.Concat(%3, %5, %7) {
            static_offsets = [[0, 0, 0, 0], [0, 0, 0, 214], [0, 0, 0, 427]]
        } : tensor<1x640x16x214x!qElemType, {order = #NWCH}>,
            tensor<1x640x16x213x!qElemType, {order = #NWCH}>,
            tensor<1x640x16x213x!qElemType, {order = #NWCH}>
                -> tensor<1x640x16x640x!qElemType, {order = #NWCH}>

        %9 = VPU.LayoutCast(%8) {
            dst_order = #NHWC
        } : tensor<1x640x16x640x!qElemType, {order = #NWCH}> -> tensor<1x640x16x640x!qElemType, {order = #NHWC}>

        %10 = VPU.AffineReshape(%9) {
            dim_mapping = [[0], [1], [2], [3]],
            shape_value = [1, 16, 640, 640]
        } : tensor<1x640x16x640x!qElemType, {order = #NHWC}> -> tensor<1x16x640x640x!qElemType, {order = #NHWC}>

        return %10 : tensor<1x16x640x640x!qElemType, {order = #NHWC}>

    }
    // CHECK:   [[SLICE_1:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 3, 214, 640] :
    // CHECK-SAME:      tensor<1x3x640x640xf16> to tensor<1x3x214x640xf16>

    // CHECK:   [[IN_RESHAPE_1:%.*]] = VPU.Reshape([[SLICE_1]]) {
    // CHECK-SAME:      shape_value = [1, 640, 3, 214]
    // CHECK-SAME:  } : tensor<1x3x214x640xf16> -> tensor<1x640x3x214xf16>

    // CHECK:   [[IN_LAYOUT_CAST_1:%.*]] = VPU.LayoutCast([[IN_RESHAPE_1]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x640x3x214xf16> -> tensor<1x640x3x214xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_QUANTIZE_1:%.*]] = VPU.NCE.PermuteQuantize([[IN_LAYOUT_CAST_1]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NWCH,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x640x16x214x!qElemType, {order = #NWCH}>

    // CHECK:   [[OUT_LAYOUT_CAST_1:%.*]] = VPU.LayoutCast([[PERMUTE_QUANTIZE_1]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x640x16x214x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x640x16x214x!qElemType, {order = #NHWC}>

    // CHECK:   [[OUT_RESHAPE_1:%.*]] = VPU.AffineReshape([[OUT_LAYOUT_CAST_1]]) {
    // CHECK-SAME:      shape_value = [1, 16, 214, 640]
    // CHECK-SAME:  } : tensor<1x640x16x214x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x214x640x!qElemType, {order = #NHWC}>


    // CHECK:   [[SLICE_2:%.*]] = VPU.Slice %arg0 [0, 0, 214, 0] [1, 3, 213, 640] :
    // CHECK-SAME:      tensor<1x3x640x640xf16> to tensor<1x3x213x640xf16>

    // CHECK:   [[IN_RESHAPE_2:%.*]] = VPU.Reshape([[SLICE_2]]) {
    // CHECK-SAME:      shape_value = [1, 640, 3, 213]
    // CHECK-SAME:  } : tensor<1x3x213x640xf16> -> tensor<1x640x3x213xf16>

    // CHECK:   [[IN_LAYOUT_CAST_2:%.*]] = VPU.LayoutCast([[IN_RESHAPE_2]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x640x3x213xf16> -> tensor<1x640x3x213xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_QUANTIZE_2:%.*]] = VPU.NCE.PermuteQuantize([[IN_LAYOUT_CAST_2]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NWCH,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x640x16x213x!qElemType, {order = #NWCH}>

    // CHECK:   [[OUT_LAYOUT_CAST_2:%.*]] = VPU.LayoutCast([[PERMUTE_QUANTIZE_2]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x640x16x213x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x640x16x213x!qElemType, {order = #NHWC}>

    // CHECK:   [[OUT_RESHAPE_2:%.*]] = VPU.AffineReshape([[OUT_LAYOUT_CAST_2]]) {
    // CHECK-SAME:      shape_value = [1, 16, 213, 640]
    // CHECK-SAME:  } : tensor<1x640x16x213x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x213x640x!qElemType, {order = #NHWC}>


    // CHECK:   [[SLICE_3:%.*]] = VPU.Slice %arg0 [0, 0, 427, 0] [1, 3, 213, 640] :
    // CHECK-SAME:      tensor<1x3x640x640xf16> to tensor<1x3x213x640xf16>

    // CHECK:   [[IN_RESHAPE_3:%.*]] = VPU.Reshape([[SLICE_3]]) {
    // CHECK-SAME:      shape_value = [1, 640, 3, 213]
    // CHECK-SAME:  } : tensor<1x3x213x640xf16> -> tensor<1x640x3x213xf16>

    // CHECK:   [[IN_LAYOUT_CAST_3:%.*]] = VPU.LayoutCast([[IN_RESHAPE_3]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x640x3x213xf16> -> tensor<1x640x3x213xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_QUANTIZE_3:%.*]] = VPU.NCE.PermuteQuantize([[IN_LAYOUT_CAST_3]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NWCH,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPETask<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x640x16x213x!qElemType, {order = #NWCH}>

    // CHECK:   [[OUT_LAYOUT_CAST_3:%.*]] = VPU.LayoutCast([[PERMUTE_QUANTIZE_3]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x640x16x213x!qElemType, {order = #NWCH}>
    // CHECK-SAME:  -> tensor<1x640x16x213x!qElemType, {order = #NHWC}>

    // CHECK:   [[OUT_RESHAPE_3:%.*]] = VPU.AffineReshape([[OUT_LAYOUT_CAST_3]]) {
    // CHECK-SAME:      shape_value = [1, 16, 213, 640]
    // CHECK-SAME:  } : tensor<1x640x16x213x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x16x213x640x!qElemType, {order = #NHWC}>

    // CHECK:   [[OUT_CONCAT:%.*]] = VPU.Concat([[OUT_RESHAPE_1]], [[OUT_RESHAPE_2]], [[OUT_RESHAPE_3]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 214, 0],
    // CHECK-SAME:          [0, 0, 427, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x16x214x640x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x16x213x640x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x16x213x640x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x640x640x!qElemType, {order = #NHWC}>

    // CHECK:   return [[OUT_CONCAT]] : tensor<1x16x640x640x!qElemType, {order = #NHWC}>

}
