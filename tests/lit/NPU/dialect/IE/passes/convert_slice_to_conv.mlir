//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-slice-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertSliceToConvFromConvert
func.func @ConvertSliceToConvFromConvert(%arg0: tensor<1x3x1088x1920xf32, {order = #NHWC}>)
    -> tensor<1x1x1088x1920xf16, {order = #NHWC}> {
    %CONVERT = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1088x1920xf32, {order = #NHWC}> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONVERT [0, 1, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16, {order = #NHWC}> to tensor<1x1x1088x1920xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x1088x1920xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<16x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C0000"> : tensor<16x48x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[CONVERT_INPUT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1088x1920xf32, {order = #NHWC}> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.ShapeCast {shape = [1, 48, 1088, 120]} inputs([[CONVERT_INPUT]] : tensor<1x3x1088x1920xf16, {order = #NHWC}>) -> tensor<1x48x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1], 
    // CHECK-SAME:      pads_begin = [0, 0], 
    // CHECK-SAME:      pads_end = [0, 0], 
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x1088x120xf16, {order = #NHWC}>, 
    // CHECK-SAME:      tensor<16x48x1x1xf16, {order = #NHWC}> 
    // CHECK-SAME:           -> tensor<1x16x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.ShapeCast {shape = [1, 1, 1088, 1920]} inputs([[CONV]] : tensor<1x16x1088x120xf16, {order = #NHWC}>) -> tensor<1x1x1088x1920xf16, {order = #NHWC}>
    
    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x1x1088x1920xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertSliceToConvFromPermuteCast
func.func @ConvertSliceToConvFromPermuteCast(%arg0: tensor<1x1088x1920x3xf16>)
    -> tensor<1x1x1088x1920xf16, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3xf16> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16, {order = #NHWC}> to tensor<1x1x1088x1920xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x1088x1920xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<16x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK:   [[PERMUTECAST_INPUT:%.*]]  = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3xf16> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.ShapeCast {shape = [1, 48, 1088, 120]} inputs([[PERMUTECAST_INPUT]] : tensor<1x3x1088x1920xf16, {order = #NHWC}>) -> tensor<1x48x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1], 
    // CHECK-SAME:      pads_begin = [0, 0], 
    // CHECK-SAME:      pads_end = [0, 0], 
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x1088x120xf16, {order = #NHWC}>, 
    // CHECK-SAME:      tensor<16x48x1x1xf16, {order = #NHWC}> 
    // CHECK-SAME:           -> tensor<1x16x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.ShapeCast {shape = [1, 1, 1088, 1920]} inputs(%2 : tensor<1x16x1088x120xf16, {order = #NHWC}>) -> tensor<1x1x1088x1920xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x1x1088x1920xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipSliceNCHW
func.func @SkipSliceNCHW(%arg0: tensor<1x3x1088x1920xf16>) -> tensor<1x1x1088x1920xf16> {
    %SLICE = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x1x1088x1920xf16>
    return %SLICE : tensor<1x1x1088x1920xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x1x1088x1920xf16>
    // CHECK:   return [[SLICE]] : tensor<1x1x1088x1920xf16>

}

// -----

// CHECK-LABEL: @SkipSliceOnHeight
func.func @SkipSliceOnHeight(%arg0: tensor<1x3x1088x1920xf16>) -> tensor<1x3x100x1920xf16> {
    %SLICE = IE.Slice %arg0 [0, 0, 3, 0] [1, 3, 100, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x3x100x1920xf16>
    return %SLICE : tensor<1x3x100x1920xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 0, 3, 0] [1, 3, 100, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x3x100x1920xf16>
    // CHECK:   return [[SLICE]] : tensor<1x3x100x1920xf16>

}

// -----

// CHECK-LABEL: @SkipSliceOnWidth
func.func @SkipSliceOnWidth(%arg0: tensor<1x3x1088x1920xf16>) -> tensor<1x3x1088x100xf16> {
    %SLICE = IE.Slice %arg0 [0, 0, 0, 3] [1, 3, 1088, 100] : tensor<1x3x1088x1920xf16> to tensor<1x3x1088x100xf16>
    return %SLICE : tensor<1x3x1088x100xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 0, 0, 3] [1, 3, 1088, 100] : tensor<1x3x1088x1920xf16> to tensor<1x3x1088x100xf16>
    // CHECK:   return [[SLICE]] : tensor<1x3x1088x100xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-LABEL: @SkipConvertSliceToConvFromPermuteCastWithQuantizedType
func.func @SkipConvertSliceToConvFromPermuteCastWithQuantizedType(%arg0: tensor<1x1088x1920x3x!qElemType>)
    -> tensor<1x1x1088x1920x!qElemType, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3x!qElemType> -> tensor<1x3x1088x1920x!qElemType, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920x!qElemType, {order = #NHWC}> to tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>

    return %SLICE : tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>

    // CHECK:   [[PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3x!qElemType> -> tensor<1x3x1088x1920x!qElemType, {order = #NHWC}>
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[PERMUTECAST]] [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920x!qElemType, {order = #NHWC}> to tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>
    // CHECK:   return [[SLICE]] : tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotConvertSliceIfChannelExpandBig
func.func @NotConvertSliceIfChannelExpandBig(%arg0: tensor<1x1088x1920x16xf16>)
    -> tensor<1x9x1088x1920xf16, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x16xf16> -> tensor<1x16x1088x1920xf16, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 9, 1088, 1920] : tensor<1x16x1088x1920xf16, {order = #NHWC}> to tensor<1x9x1088x1920xf16, {order = #NHWC}>

    return %SLICE : tensor<1x9x1088x1920xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTECAST_INPUT:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x16xf16> -> tensor<1x16x1088x1920xf16, {order = #NHWC}>
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[PERMUTECAST_INPUT]] [0, 3, 0, 0] [1, 9, 1088, 1920] : tensor<1x16x1088x1920xf16, {order = #NHWC}> to tensor<1x9x1088x1920xf16, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x9x1088x1920xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertIfSliceNotFromDDR
func.func @NotConvertIfSliceNotFromDDR(%arg0: tensor<1x4x1088x1920xf16, {order = #NHWC}>, %arg1: tensor<1x4x1088x1920xf16, {order = #NHWC}>)
        -> tensor<1x3x1088x1920xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 1, 0, 0] [1, 3, 1088, 1920] : tensor<1x4x1088x1920xf16, {order = #NHWC}> to tensor<1x3x1088x1920xf16, {order = #NHWC}>

    return %0 : tensor<1x3x1088x1920xf16, {order = #NHWC}>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 3, 1088, 1920] : tensor<1x4x1088x1920xf16, {order = #NHWC}> to tensor<1x3x1088x1920xf16, {order = #NHWC}>
    // CHECK:   return [[SLICE]] : tensor<1x3x1088x1920xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotConvertIfFitCmx
func.func @NotConvertIfFitCmx(%arg0: tensor<1x16x16x3xf16>)
    -> tensor<1x1x16x16xf16, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x3xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 1, 16, 16] : tensor<1x3x16x16xf16, {order = #NHWC}> to tensor<1x1x16x16xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x16x16xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x3xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[PERMUTECAST]] [0, 3, 0, 0] [1, 1, 16, 16] : tensor<1x3x16x16xf16, {order = #NHWC}> to tensor<1x1x16x16xf16, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x1x16x16xf16, {order = #NHWC}>
}
