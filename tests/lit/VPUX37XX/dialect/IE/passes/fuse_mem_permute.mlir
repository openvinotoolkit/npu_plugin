//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --fuse-mem-permute  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteNWHC
func.func @MemPermuteNWHC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x64x32xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x64x32xf16>

    return %1 : tensor<1x16x64x32xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NCHW}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 16, 64, 32]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16>)
    // CHECK-SAME:  -> tensor<1x16x64x32xf16>
    // CHECK:   return [[RESHAPE]] : tensor<1x16x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCH
func.func @MemPermuteNWCH(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x32x64xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NWCH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16>

    return %1 : tensor<1x16x32x64xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   return [[CONV]] : tensor<1x16x32x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteNHCW
func.func @MemPermuteNHCW(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x32x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NHCW
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x32x16xf16>

    return %1 : tensor<1x64x32x16xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWHC}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NCHW}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 64, 32, 16]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16>)
    // CHECK-SAME:  -> tensor<1x64x32x16xf16>
    // CHECK:   return [[RESHAPE]] : tensor<1x64x32x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteNHWC
func.func @MemPermuteNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x16x32xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x16x32xf16>

    return %1 : tensor<1x64x16x32xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWCH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NCHW}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 64, 16, 32]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16>)
    // CHECK-SAME:  -> tensor<1x64x16x32xf16>
    // CHECK:   return [[RESHAPE]] : tensor<1x64x16x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MemPermuteNCWH
func.func @MemPermuteNCWH(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x32x16x64xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NCHW,
        mem_perm = #NCWH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16>

    return %1 : tensor<1x32x16x64xf16>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NHCW}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NCHW}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 32, 16, 64]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16>)
    // CHECK-SAME:  -> tensor<1x32x16x64xf16>
    // CHECK:   return [[RESHAPE]] : tensor<1x32x16x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteNWHCOrderNHWC
func.func @MemPermuteNWHCOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x32x16x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16, {order = #NHWC}>

    return %1 : tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NHWC}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 32, 16, 64]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x32x16x64xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE]] : tensor<1x32x16x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCHOrderNHWC
func.func @MemPermuteNWCHOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x16x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NWCH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x16x32xf16, {order = #NHWC}>

    return %1 : tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NHWC}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 64, 16, 32]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x16x32xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE]] : tensor<1x64x16x32xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteNHCWOrderNHWC
func.func @MemPermuteNHCWOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x16x64x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHCW
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x64x32xf16, {order = #NHWC}>

    return %1 : tensor<1x16x64x32xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWHC}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NHWC}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 16, 64, 32]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x16x64x32xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE]] : tensor<1x16x64x32xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteNHWCOrderNHWC
func.func @MemPermuteNHWCOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x32x64x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16, {order = #NHWC}>

    return %1 : tensor<1x32x64x16xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NWCH}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NHWC}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 32, 64, 16]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x32x64x16xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE]] : tensor<1x32x64x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MemPermuteNCWHOrderNHWC
func.func @MemPermuteNCWHOrderNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> tensor<1x64x32x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}>,
        tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x64x32x16xf16, {order = #NHWC}>

    return %1 : tensor<1x64x32x16xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution
    // CHECK-SAME: -> tensor<1x16x32x64xf16, {order = #NHCW}>
    // CHECK-NOT: IE.MemPermute

    // CHECK:   [[LAYOUT:%.*]] = IE.LayoutCast([[CONV]]) {dst_order = #NHWC}
    // CHECK:   [[RESHAPE:%.*]] = IE.ShapeCast {shape = [1, 64, 32, 16]}
    // CHECK-SAME:  inputs([[LAYOUT]] : tensor<1x16x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:  -> tensor<1x64x32x16xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE]] : tensor<1x64x32x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @MultiplyMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.Multiply(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.Multiply(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SubtractMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @SubtractMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.Subtract(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.Subtract(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AndMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @AndMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.And(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.And(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}
