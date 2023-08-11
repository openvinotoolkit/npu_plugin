//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-through-add %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertAddNWCH
func.func @ConvertAddNWCH(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x8x4x76xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x4x76xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 76]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4x76xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ConvertAddNWHC
func.func @ConvertAddNWHC(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x8x76x4xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWHC
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x76x4xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x76x4xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x76x4xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x76x4xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 76, 4]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x76x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddNCWH
func.func @ConvertAddNCWH(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x8x76xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NCWH
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x76xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 4, 8, 76]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x4x8x76xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddNCHW
func.func @ConvertAddNCHW(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x76x8xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NCHW
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x4x76x8xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x76x8xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x76x8xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x76x8xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 4, 76, 8]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x4x76x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertAddNHCW
func.func @ConvertAddNHCW(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x76x4x8xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NHCW
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x76x4x8xf16>

    return %OUT_MEM_PERMUTE : tensor<1x76x4x8xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NHCW
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x76x4x8xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NHCW
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x76x4x8xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 76, 4, 8]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x76x4x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertAddWithPostOp
func.func @ConvertAddWithPostOp(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x8x4x76xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        post_op = {attrs = {negative_slope = 0.1}, name = "IE.LeakyRelu"}
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x4x76xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 76]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4x76xf16>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddWithQuantizeCast
func.func @ConvertPermuteAddWithQuantizeCast(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType0> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%LHS_MEM_PERMUTE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%RHS_MEM_PERMUTE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs(%ADD : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x16x4096x2048xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x4096x2048xf16> -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x16x4096x2048xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x4096x2048xf16> -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x4096x2048x!qElemType1>) -> tensor<1x8x4096x4096x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1> -> tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x4096x4096x!qElemType0>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddWithQuantizeCastNoShapeCast
func.func @ConvertPermuteAddWithQuantizeCastNoShapeCast(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType0> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType1>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x8x4096x4096x!qElemType1>) -> tensor<1x8x4096x4096x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1> -> tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x4096x4096x!qElemType0>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddNoShapeCast
func.func @ConvertPermuteAddNoShapeCast(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x8x4096x4096x!qElemType>) -> tensor<1x8x4096x4096x!qElemType>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4096x4096x!qElemType>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithQuantizeCast
func.func @ConvertPermuteQuantizeAddWithQuantizeCast(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType0> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs(%ADD : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x16x4096x2048xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x4096x2048xf16> -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x16x4096x2048xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x4096x2048xf16> -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x4096x2048x!qElemType1>) -> tensor<1x8x4096x4096x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1> -> tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x4096x4096x!qElemType0>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPopagateIfPemutationsCanNotFold
func.func @NoPopagateIfPemutationsCanNotFold(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs(%ADD : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[LHS_PERMUTEQUANTIZE]] : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[RHS_PERMUTEQUANTIZE]] : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_SHAPE_CAST]], [[RHS_SHAPE_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[ADD]] : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    // CHECK:   [[OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[OUT_QUANTIZE_CAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    // CHECK:   return [[OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAdd
func.func @ConvertPermuteQuantizeAdd(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs(%ADD : tensor<1x16x4096x2048x!qElemType, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x16x4096x2048xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x4096x2048xf16> -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x16x4096x2048xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x4096x2048xf16> -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x4096x2048x!qElemType, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x4096x2048x!qElemType>) -> tensor<1x8x4096x4096x!qElemType>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4096x4096x!qElemType>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithQuantizeCastNoShapeCast
func.func @ConvertPermuteQuantizeAddWithQuantizeCastNoShapeCast(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType0> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_PERMUTEQUANTIZE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType0, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType0>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType1>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x8x4096x4096x!qElemType1>) -> tensor<1x8x4096x4096x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType0} : tensor<1x8x4096x4096x!qElemType1> -> tensor<1x8x4096x4096x!qElemType0>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x4096x4096x!qElemType0>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddNoShapeCast
func.func @ConvertPermuteQuantizeAddNoShapeCast(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_PERMUTEQUANTIZE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x4096x4096xf16, {order = #NHWC}>, tensor<1x8x4096x4096xf16, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x8x4096x4096x!qElemType>) -> tensor<1x8x4096x4096x!qElemType>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4096x4096x!qElemType>
}
