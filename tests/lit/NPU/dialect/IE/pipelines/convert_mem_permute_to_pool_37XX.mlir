//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-mem-permute-to-pool %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL: @MemPermuteNHCWInNCHWOutNHCWPerm
func.func @MemPermuteNHCWInNCHWOutNHCWPerm(%arg0: tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>}>)
        -> tensor<1x32x48x64xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
    } : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>}>
        -> tensor<1x32x48x64xf16>

    return %MEM_PERMUTE : tensor<1x32x48x64xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 48, 32]
    // CHECK-SAME:  } inputs(%arg0 : tensor<1x32x48x64xf16, {order = #NHCW}>)
    // CHECK-SAME:      -> tensor<1x64x48x32xf16, {order = #NHCW}>

    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x64x48x32xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x64x48x32xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_LAYOUT_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x48x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x48x32xf16, {order = #NWHC}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x64x48x32xf16, {order = #NWHC}> -> tensor<1x64x48x32xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 32, 48, 64]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x64x48x32xf16>) -> tensor<1x32x48x64xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x32x48x64xf16>
}

// -----

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNHWCPerm
func.func @MemPermuteNCHWInNCHWOutNHWCPerm(%arg0: tensor<1x32x48x64xf16>)
        -> tensor<1x48x64x32xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    } : tensor<1x32x48x64xf16> -> tensor<1x48x64x32xf16>

    return %MEM_PERMUTE : tensor<1x48x64x32xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 32, 48]
    // CHECK-SAME:  } inputs(%arg0 : tensor<1x32x48x64xf16>)
    // CHECK-SAME:      -> tensor<1x64x32x48xf16>

    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x64x32x48xf16>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_LAYOUT_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NWCH}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NWCH}> -> tensor<1x64x32x48xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 64, 32]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x64x32x48xf16>) -> tensor<1x48x64x32xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x48x64x32xf16>
}

// -----

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNHCWPerm
func.func @MemPermuteNCHWInNCHWOutNHCWPerm(%arg0: tensor<1x32x48x64xf16>)
        -> tensor<1x48x32x64xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
    } : tensor<1x32x48x64xf16> -> tensor<1x48x32x64xf16>

    return %MEM_PERMUTE : tensor<1x48x32x64xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 32, 48]
    // CHECK-SAME:  } inputs(%arg0 : tensor<1x32x48x64xf16>)
    // CHECK-SAME:      -> tensor<1x64x32x48xf16>

    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x64x32x48xf16>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_LAYOUT_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NWHC}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NWHC}> -> tensor<1x64x32x48xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 32, 64]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x64x32x48xf16>) -> tensor<1x48x32x64xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x48x32x64xf16>
}

// -----

// CHECK-LABEL: @MemPermuteNCWHInNHWCOutNWHCPerm
func.func @MemPermuteNCWHInNHWCOutNWHCPerm(%arg0: tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}>)
        -> tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
    } : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}>
        -> tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %MEM_PERMUTE : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 48, 32, 64]
    // CHECK-SAME:  } inputs(%arg0 : tensor<1x32x48x64xf16, {order = #NCWH}>)
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NCWH}>

    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_LAYOUT_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NCWH}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NHWC}>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 32, 48, 64]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x48x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x32x48x64xf16, {order = #NHWC}>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x32x48x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNCWHPerm
func.func @MemPermuteNCHWInNCHWOutNCWHPerm(%arg0: tensor<1x8x1500x64xf16>) -> tensor<1x8x64x1500xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
    } : tensor<1x8x1500x64xf16> -> tensor<1x8x64x1500xf16>

    return %MEM_PERMUTE : tensor<1x8x64x1500xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 8, 1500]
    // CHECK-SAME:  } inputs(%arg0 : tensor<1x8x1500x64xf16>)
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16>

    // CHECK:       [[IN_LAYOUT_CAST:%.*]] = IE.LayoutCast([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.*]] = IE.MaxPool([[IN_LAYOUT_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16, {order = #NHCW}>

    // CHECK:       [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16>

    // CHECK:       [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 64, 1500]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x64x8x1500xf16>)
    // CHECK-SAME:      -> tensor<1x8x64x1500xf16>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x64x1500xf16>
}

// -----

// CHECK-LABEL: @SkipMemPermuteWithMisalignedShape
func.func @SkipMemPermuteWithMisalignedShape(%arg0: tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}>)
        -> tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
    } : tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}> -> tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %MEM_PERMUTE : tensor<1x32x47x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
}

// -----

// CHECK-LABEL: @SkipTrivialMemPermute
func.func @SkipTrivialMemPermute(%arg0: tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>)
        -> tensor<1x48x64x32xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    } : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        -> tensor<1x48x64x32xf16>

    return %MEM_PERMUTE : tensor<1x48x64x32xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @UnsupportedDstOrder
func.func @UnsupportedDstOrder(%arg0: tensor<1x32x48x64xf16, {order = #NCWH}>)
        -> tensor<1x64x48x32xf16, {order = #NHCW}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHCW,
        mem_perm = #NWHC
    } : tensor<1x32x48x64xf16, {order = #NCWH}> -> tensor<1x64x48x32xf16, {order = #NHCW}>

    return %MEM_PERMUTE : tensor<1x64x48x32xf16, {order = #NHCW}>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnsupportedDimN
func.func @UnsupportedDimN(%arg0: tensor<25x14x14x2304xf16>) -> tensor<25x14x2304x14xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>

    return %MEM_PERMUTE : tensor<25x14x2304x14xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} :
    // CHECK-SAME:  tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<25x14x2304x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithAlignChannel(%arg0: tensor<1x32x255x511xf16, {order = #NHWC}>) -> tensor<1x32x255x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x32x255x511xf16, {order = #NHWC}> -> tensor<1x32x255x511xf16>
    return %MEM_PERMUTE : tensor<1x32x255x511xf16>

    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:        kernel_size = [1, 1],
    // CHECK-SAME:        pads_begin = [0, 0],
    // CHECK-SAME:        pads_end = [0, 0],
    // CHECK-SAME:        rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:        strides = [1, 1]} : tensor<1x32x255x511xf16, {order = #NHWC}> -> tensor<1x32x255x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x32x255x511xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithUnalignedChannel
func.func @MemPermuteNHWCInNCHWOutWithUnalignedChannel(%arg0: tensor<1x3x256x512xf16, {order = #NHWC}>) -> tensor<1x3x256x512xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x512xf16, {order = #NHWC}> -> tensor<1x3x256x512xf16>
    return %MEM_PERMUTE : tensor<1x3x256x512xf16>

    // CHECK:       [[SHAPE_CAST_WC_IN_W:%.*]] = IE.ShapeCast {shape = [1, 1, 256, 1536]} inputs(%arg0 : tensor<1x3x256x512xf16, {order = #NHWC}>) -> tensor<1x1x256x1536xf16, {order = #NHWC}>
    // CHECK:       [[CAST_WC_NCHW:%.*]] = IE.LayoutCast([[SHAPE_CAST_WC_IN_W]]) {dst_order = #NCHW} : tensor<1x1x256x1536xf16, {order = #NHWC}> -> tensor<1x1x256x1536xf16>
    // CHECK:       [[CAST_WC_IN_C:%.*]] = IE.ShapeCast {shape = [1, 1536, 1, 256]} inputs([[CAST_WC_NCHW]] : tensor<1x1x256x1536xf16>) -> tensor<1x1536x1x256xf16>
    // CHECK:       [[LAYOUT_CAST_0_NHWC:%.*]] = IE.LayoutCast([[CAST_WC_IN_C]]) {dst_order = #NHWC} : tensor<1x1536x1x256xf16> -> tensor<1x1536x1x256xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.*]] = IE.MaxPool([[LAYOUT_CAST_0_NHWC]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x1536x1x256xf16, {order = #NHWC}> -> tensor<1x1536x1x256xf16, {order = #NHCW}>
    // CHECK:       [[LAYOUT_CAST_0_NCHW:%.*]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NCHW} : tensor<1x1536x1x256xf16, {order = #NHCW}> -> tensor<1x1536x1x256xf16>
    // CHECK:       [[SHAPE_CAST_WC_IN_H:%.*]] = IE.ShapeCast {shape = [1, 1, 1536, 256]} inputs([[LAYOUT_CAST_0_NCHW]] : tensor<1x1536x1x256xf16>) -> tensor<1x1x1536x256xf16>
    // CHECK:       [[SHAPE_CAST_CH_IN_W:%.*]] = IE.ShapeCast {shape = [1, 1, 512, 768]} inputs([[SHAPE_CAST_WC_IN_H]] : tensor<1x1x1536x256xf16>) -> tensor<1x1x512x768xf16>
    // CHECK:       [[SHAPE_CAST_CH_IN_C:%.*]] = IE.ShapeCast {shape = [1, 768, 1, 512]} inputs([[SHAPE_CAST_CH_IN_W]] : tensor<1x1x512x768xf16>) -> tensor<1x768x1x512xf16>
    // CHECK:       [[LAYOUT_CAST_1_NHWC:%.*]] = IE.LayoutCast([[SHAPE_CAST_CH_IN_C]]) {dst_order = #NHWC} : tensor<1x768x1x512xf16> -> tensor<1x768x1x512xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.*]] = IE.MaxPool([[LAYOUT_CAST_1_NHWC]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x768x1x512xf16, {order = #NHWC}> -> tensor<1x768x1x512xf16, {order = #NHCW}>
    // CHECK:       [[LAYOUT_CAST_1_NCHW:%.*]] = IE.LayoutCast([[MAXPOOL_1]]) {dst_order = #NCHW} : tensor<1x768x1x512xf16, {order = #NHCW}> -> tensor<1x768x1x512xf16>
    // CHECK:       [[SHAPE_CAST_CH_IN_H:%.*]] = IE.ShapeCast {shape = [1, 1, 768, 512]} inputs([[LAYOUT_CAST_1_NCHW]] : tensor<1x768x1x512xf16>) -> tensor<1x1x768x512xf16>
    // CHECK:       [[RESULT:%.*]] = IE.ShapeCast {shape = [1, 3, 256, 512]} inputs([[SHAPE_CAST_CH_IN_H]] : tensor<1x1x768x512xf16>) -> tensor<1x3x256x512xf16>
    // CHECK:       return [[RESULT]] : tensor<1x3x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithHCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithHCNotAlignChannel(%arg0: tensor<1x3x255x512xf16, {order = #NHWC}>) -> tensor<1x3x255x512xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x255x512xf16, {order = #NHWC}> -> tensor<1x3x255x512xf16>
    return %MEM_PERMUTE : tensor<1x3x255x512xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x255x512xf16, {order = #NHWC}> -> tensor<1x3x255x512xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x255x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel(%arg0: tensor<1x3x256x511xf16, {order = #NHWC}>) -> tensor<1x3x256x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    return %MEM_PERMUTE : tensor<1x3x256x511xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x511xf16>
}
