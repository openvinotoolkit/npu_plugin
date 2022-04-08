//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-reorders %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubView
module @ReorderWithSubView attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16>)
func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x4x4x2xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 4, 4, 2] : tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x4x4x2xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x4x4x2xf16, {order = #NHWC}> -> tensor<1x4x4x2xf16>
    return %2 : tensor<1x4x4x2xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x8x4x2xf16> to tensor<1x4x4x2xf16>
    // CHECK:       return [[VAR0]] : tensor<1x4x4x2xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithTwoUsersSubView
module @ReorderWithTwoUsersSubView attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x8x4x2xf16, {order = #NHWC}>) -> (tensor<1x4x4x2xf16, {order = #NHWC}>, tensor<1x5x4x2xf16, {order = #NHWC}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHWC}> -> tensor<1x8x4x2xf16>
    %1 = IE.Slice %0 [0, 2, 0, 0] [1, 4, 4, 2] : tensor<1x8x4x2xf16> to tensor<1x4x4x2xf16>
    %2 = IE.Slice %0 [0, 2, 0, 0] [1, 5, 4, 2] : tensor<1x8x4x2xf16> to tensor<1x5x4x2xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x4x4x2xf16> -> tensor<1x4x4x2xf16, {order = #NHWC}>
    %4 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x5x4x2xf16> -> tensor<1x5x4x2xf16, {order = #NHWC}>
    return %3, %4 : tensor<1x4x4x2xf16, {order = #NHWC}>, tensor<1x5x4x2xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x4x4x2xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x5x4x2xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]], [[VAR1]] : tensor<1x4x4x2xf16, {order = #NHWC}>, tensor<1x5x4x2xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandNeedSwap
module @ReorderWithExpandNeedSwap attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK:       func @main([[ARG0:%arg[0-9]+]]: tensor<1x5x512x512xf16>)
func @main(%arg0: tensor<1x5x512x512xf16>) -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x5x512x512xf16> -> tensor<1x5x512x512xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %3 = IE.Convolution(%2, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = {attrs = {}, name = "IE.ReLU"}, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %3 : tensor<1x16x512x512xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR1:%.*]] = IE.Expand(%arg0)
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 11, 0, 0]
    // CHECK-SAME:      : tensor<1x5x512x512xf16> -> tensor<1x16x512x512xf16>

    // CHECK:       [[VAR2:%.*]] = IE.Reorder([[VAR1]])
    // CHECK-SAME:      tensor<1x16x512x512xf16> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.*]] = IE.Convolution([[VAR2]], [[VAR0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       return [[VAR3]] : tensor<1x16x512x512xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandNoSwap
module @ReorderWithExpandNoSwap attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK:       func @main([[ARG0:%arg[0-9]+]]: tensor<1x1x512x512xf16>)
func @main(%arg0: tensor<1x1x512x512xf16>) -> tensor<1x16x512x512xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %3 = IE.Convolution(%2, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = {attrs = {}, name = "IE.ReLU"}, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %3 : tensor<1x16x512x512xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR1:%.*]] = IE.Reorder(%arg0)
    // CHECK-SAME:      tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.*]] = IE.Expand([[VAR1]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      : tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.*]] = IE.Convolution([[VAR2]], [[VAR0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       return [[VAR3]] : tensor<1x16x512x512xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpand
module @ReorderWithExpand attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x13xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>

    %1 = IE.Expand(%0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x30x30xf16> -> tensor<1x16x30x30xf16>

    %2 = IE.MaxPool(%1) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = "FLOOR",
        strides = [2, 2]
    } : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>

    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 3, 15, 13] : tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x3x15x13xf16> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    return %4 : tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Expand([[ARG0]]
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR1:%.+]] = IE.MaxPool([[VAR0]])
    // CHECK-SAME:      tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x15x13xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.+]] = IE.Slice [[VAR1]]
    // CHECK-SAME:      tensor<1x16x15x13xf16, {order = #NHWC}> to tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK        return [[VAR2]] : tensor<1x3x15x13xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127, 0.0013264333169291339:127,9.8713551919291337E-4:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127,0.0013264333169291339:127}>
!qElemType3 = type !quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127, 0.0013264333169291339:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127}>

module @ReorderWithQuantExpandAndSlice attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30x!qElemType0>)
func @main(%arg0: tensor<1x3x30x30x!qElemType0>) -> tensor<1x3x15x13x!qElemType1> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x30x30x!qElemType0> -> tensor<1x3x30x30x!qElemType0, {order = #NHWC}>

    %1 = IE.Expand(%0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x30x30x!qElemType0, {order = #NHWC}> -> tensor<1x16x30x30x!qElemType2, {order = #NHWC}>

    %2 = IE.MaxPool(%1) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = "FLOOR",
        strides = [2, 2]
    } : tensor<1x16x30x30x!qElemType2, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType3, {order = #NHWC}>

    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x16x15x13x!qElemType3, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType3>

    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 3, 15, 13] : tensor<1x16x15x13x!qElemType3> to tensor<1x3x15x13x!qElemType1>

    return %4 : tensor<1x3x15x13x!qElemType1>

    // CHECK: [[VAR0:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
    // CHECK-SAME:     tensor<1x3x30x30x!qElemType0> ->
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType2>

    // CHECK: [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} :
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType2> ->
    // CHECK-SAME:     tensor<1x16x30x30x!qElemType2, {order = #NHWC}>

    // CHECK: [[VAR2:%.+]] = IE.MaxPool([[VAR1]])

    // CHECK: [[VAR3:%.+]] = IE.Slice [[VAR2]] [0, 0, 0, 0] [1, 3, 15, 13] :
    // CHECK-SAME:     tensor<1x16x15x13x!qElemType3, {order = #NHWC}> to
    // CHECK-SAME:     tensor<1x3x15x13x!qElemType1, {order = #NHWC}>

    // CHECK: [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NCHW} :
    // CHECK-SAME:     tensor<1x3x15x13x!qElemType1, {order = #NHWC}> ->
    // CHECK-SAME:     tensor<1x3x15x13x!qElemType1>

    // CHECK: return [[VAR4]] : tensor<1x3x15x13x!qElemType1>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSplit
module @ReorderWithSplit attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) ->
        (tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>

    %1:3 = IE.Split(%0) {axis_value = 1, num_splits = 3} :
        tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

    %2 = IE.Reorder(%1#0) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1#1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    %4 = IE.Reorder(%1#2) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>

    return %2, %3, %4 : tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%[0-9]+]]:3 = IE.Split([[ARG0]])
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> ->
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR0]]#0, [[VAR0]]#1, [[VAR0]]#2
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSplitMultipleUses
module @ReorderWithSplitMultipleUses attributes {VPUIP.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) ->
        (tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>){
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>

    %1:3 = IE.Split(%0) {axis_value = 1, num_splits = 3} :
        tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

    %2 = IE.Reorder(%1#1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1#1) {dstOrder = #NHWC} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%[0-9]+]]:3 = IE.Split([[ARG0]])
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> ->
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:          tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[VAR0]]#1, [[VAR0]]#1
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithConcat
module @ReorderWithConcat attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {per_axis = {axis = 1}} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x2x30x30xf16> -> tensor<1x2x30x30xf16, {order = #NHWC}>
    return %3 : tensor<1x2x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[ARG1]]) {per_axis = {axis = 1 : i64}}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x2x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithConcatWithConsts
module @ReorderWithConcatWithConsts attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = const.Declare tensor<1x1x30x30xf16> = dense<1.0> : tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {per_axis = {axis = 1}} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x2x30x30xf16> -> tensor<1x2x30x30xf16, {order = #NHWC}>
    return %3 : tensor<1x2x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CST0:%.*]] = const.Declare
    // CHECK-SAME:      #const.Reorder<#NHWC>
    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[CST0]]) {per_axis = {axis = 1 : i64}}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x2x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderPropagationWithConcat
module @ReorderPropagationWithConcat attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x29x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {per_axis = {axis = 1}} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Slice %2 [0, 0, 1, 0] [1, 2, 29, 30] : tensor<1x2x30x30xf16> to tensor<1x2x29x30xf16>
    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x2x29x30xf16> -> tensor<1x2x29x30xf16, {order = #NHWC}>
    return %4 : tensor<1x2x29x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[ARG1]]) {per_axis = {axis = 1 : i64}}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Slice [[VAR0]]
    // CHECK-SAME:      to tensor<1x2x29x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR1]] : tensor<1x2x29x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandTwoBranches
module @ReorderWithExpandTwoBranches attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK:       func @main([[ARG0:%arg[0-9]+]]: tensor<1x24x56x56xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x24x56x56xf16, {order = #NHWC}>) -> tensor<1x32x56x56xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x24x56x56xf16, {order = #NHWC}> -> tensor<1x24x56x56xf16>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x56x56xf16> -> tensor<1x32x56x56xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x32x56x56xf16> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %4 = IE.Convolution(%3, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = {attrs = {}, name = "IE.ReLU"}, strides = [1, 1]} : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %5 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x56x56xf16> -> tensor<1x32x56x56xf16>
    %6 = IE.Reorder(%5) {dstOrder = #NHWC} : tensor<1x32x56x56xf16> -> tensor<1x32x56x56xf16, {order = #NHWC}>
    %7 = IE.Add(%6, %4) {auto_broadcast = "NUMPY"} : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<1x32x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>

    return %7 : tensor<1x32x56x56xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.*]] = const.Declare
    // CHECK-SAME:      tensor<32x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR1:%.*]] = IE.Expand([[ARG0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 8, 0, 0]
    // CHECK-SAME:      : tensor<1x24x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.*]] = IE.Convolution([[VAR1]], [[VAR0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      post_op = {attrs = {}, name = "IE.ReLU"}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x56x56xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.*]] = IE.Expand([[ARG0]])
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end = [0, 8, 0, 0]
    // CHECK-SAME:      : tensor<1x24x56x56xf16, {order = #NHWC}> -> tensor<1x32x56x56xf16, {order = #NHWC}>


    // CHECK:       [[VAR4:%.*]] = IE.Add([[VAR3]], [[VAR2]])
    // CHECK-SAME:      {auto_broadcast = "NUMPY"}
    // CHECK-SAME:      : tensor<1x32x56x56xf16, {order = #NHWC}>, tensor<1x32x56x56xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x56x56xf16, {order = #NHWC}>

    // CHECK:       return [[VAR4]] : tensor<1x32x56x56xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithLayer
module @ReorderWithLayer attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16>
    %1 = IE.SoftMax(%0) { axisInd = 1 } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.SoftMax([[ARG0]]
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x30x30xf16, {order = #NHWC}>

    // CHECK        return [[VAR0]] : tensor<1x3x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @ReorderWithQuantizeCast
module @ReorderWithQuantizeCast attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%.+]]: tensor<1x3x30x30xui8, {order = #NHWC}>)
func @main(%arg0: tensor<1x3x30x30xui8, {order = #NHWC}>) -> tensor<1x3x30x30x!qElemType, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x30x30xui8, {order = #NHWC}> -> tensor<1x3x30x30xui8>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x3x30x30xui8> -> tensor<1x3x30x30x!qElemType>

    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>
    %3 = IE.And(%2, %2) {auto_broadcast = "NONE_OR_EXPLICIT"} :
            tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<1x3x30x30x!qElemType, {order = #NHWC}>
            -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    // CHECK-NOT:  IE.Reorder

    // CHECK:      [[VAR0:%.+]] = IE.QuantizeCast([[ARG0:%.+]]) {dstElemType = !qElemType} :
    // CHECK-SAME:     tensor<1x3x30x30xui8, {order = #NHWC}> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    // CHECK-NOT:  IE.Reorder

    // CHECK:      [[VAR1:%.+]] = IE.And([[VAR0]], [[VAR0]]) {auto_broadcast = "NONE_OR_EXPLICIT"} :
    // CHECK-SAME:     tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x3x30x30x!qElemType, {order = #NHWC}>

    // CHECK:      return [[VAR1]] : tensor<1x3x30x30x!qElemType, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.51323526419845278:128>
!qElemType1 = type !quant.uniform<u8:f16, 0.25661763209922639:128>
!qElemType2 = type !quant.uniform<u8:f16, 0.12830881604961319:128>

module @ReorderWithQuantizeCastTwoBranches attributes {VPU.compilationMode = "DefaultHW"} {

func @main(%arg0: tensor<1x48x14x14x!qElemType0, {order = #NHWC}>) -> (tensor<1x48x14x14x!qElemType2, {order = #NHWC}>, tensor<1x14x14x40x!qElemType1>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x48x14x14x!qElemType0, {order = #NHWC}> -> tensor<1x48x14x14x!qElemType0>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 40, 14, 14] : tensor<1x48x14x14x!qElemType0> to tensor<1x40x14x14x!qElemType0>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<1x40x14x14x!qElemType0> -> tensor<1x40x14x14x!qElemType1>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType2} : tensor<1x40x14x14x!qElemType1> -> tensor<1x40x14x14x!qElemType2>
    %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x40x14x14x!qElemType2> -> tensor<1x48x14x14x!qElemType2>
    %5 = IE.Reorder(%4) {dstOrder = #NHWC} : tensor<1x48x14x14x!qElemType2> -> tensor<1x48x14x14x!qElemType2, {order = #NHWC}>
    %6 = IE.Reshape(%2) {shape_value = [1, 14, 14, 40]} : tensor<1x40x14x14x!qElemType1> -> tensor<1x14x14x40x!qElemType1>

   return %5, %6 : tensor<1x48x14x14x!qElemType2, {order = #NHWC}>, tensor<1x14x14x40x!qElemType1>

    // CHECK-NOT:  IE.Reorder
    // CHECK:      [[SLICE:%.+]] = IE.Slice %arg0
    // CHECK-SAME:  tensor<1x48x14x14x!qElemType0, {order = #NHWC}> to tensor<1x40x14x14x!qElemType0, {order = #NHWC}>
    // CHECK:      [[QCAST0:%.+]] = IE.QuantizeCast([[SLICE]]) {dstElemType = !qElemType2}
    // CHECK-SAME: tensor<1x40x14x14x!qElemType0, {order = #NHWC}> -> tensor<1x40x14x14x!qElemType2, {order = #NHWC}>
    // CHECK:      [[REORDER:%.+]] = IE.Reorder([[QCAST0]]) {dstOrder = #NCHW}
    // CHECK-SAME: tensor<1x40x14x14x!qElemType2, {order = #NHWC}> -> tensor<1x40x14x14x!qElemType2>
    // CHECK:      [[QCAST1:%.+]] = IE.QuantizeCast([[QCAST0]]) {dstElemType = !qElemType1}
    // CHECK-SAME: tensor<1x40x14x14x!qElemType2, {order = #NHWC}> -> tensor<1x40x14x14x!qElemType1, {order = #NHWC}>
    // CHECK:      [[RESULT0:%.+]] = IE.Expand([[QCAST1]])
    // CHECK-SAME: tensor<1x40x14x14x!qElemType1, {order = #NHWC}> -> tensor<1x48x14x14x!qElemType1, {order = #NHWC}>
    // CHECK:      [[RESULT1:%.+]] = IE.Reshape([[REORDER]])
    // CHECK-SAME: tensor<1x40x14x14x!qElemType2> -> tensor<1x14x14x40x!qElemType2>
    // CHECK:      return [[RESULT0]], [[RESULT1]] : tensor<1x48x14x14x!qElemType1, {order = #NHWC}>, tensor<1x14x14x40x!qElemType2>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CMajorToZMajorConv
module @CMajorToZMajorConv attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
func @main(%arg0: tensor<1x3x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x32x32xf16, {order = #NHWC}> -> tensor<1x3x32x32xf16>

    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x32x32xf16>, tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    return %1 : tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}>
    // CHECK:       [[VAR0:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK-SAME:       -> tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x16x32x32xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>

// CHECK-LABEL: @ReorderPermuteCastChainWithSlice
module @ReorderPermuteCastChainWithSlice attributes {VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%.+]]: tensor<1x128x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x128x30x30xf16, {order = #NHWC}>) -> (tensor<1x4x30x30x16xf16, {order = #NCDHW}>, tensor<1x4x30x30x16xf16, {order = #NCDHW}>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x128x30x30xf16, {order = #NHWC}> -> tensor<1x128x30x30xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 4, 32, 30, 30]} :
            tensor<1x128x30x30xf16> -> tensor<1x4x32x30x30xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #map1, mem_perm = #NCDHW} : tensor<1x4x32x30x30xf16> -> tensor<1x4x30x30x32xf16, {order = #map1}>
    %3 = IE.Reorder(%2) {dstOrder = #NCDHW} : tensor<1x4x30x30x32xf16, {order = #map1}> -> tensor<1x4x30x30x32xf16, {order = #NCDHW}>
    %4 = IE.Slice %3 [0, 0, 0, 0, 0] [1, 4, 30, 30, 16] : tensor<1x4x30x30x32xf16, {order = #NCDHW}> to tensor<1x4x30x30x16xf16, {order = #NCDHW}>
    %5 = IE.Slice %3 [0, 0, 0, 0, 1] [1, 4, 30, 30, 16] : tensor<1x4x30x30x32xf16, {order = #NCDHW}> to tensor<1x4x30x30x16xf16, {order = #NCDHW}>
    return %4, %5 : tensor<1x4x30x30x16xf16, {order = #NCDHW}>, tensor<1x4x30x30x16xf16, {order = #NCDHW}>

    // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME(LITERAL):       {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [1, 4, 32, 30, 30]}
    // CHECK-SAME:      -> tensor<1x4x32x30x30xf16, {order = #map0}>
    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[AFFINERESHAPE]]) {dstOrder = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x32x30x30xf16>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[REORDER0]]) {dst_order = #map1, mem_perm = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x30x30x32xf16, {order = #map1}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[PERMUTECAST]]) {dstOrder = #NCDHW}
    // CHECK-SAME:      -> tensor<1x4x30x30x32xf16, {order = #NCDHW}>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[REORDER1]]
    // CHECK-SAME:      tensor<1x4x30x30x32xf16, {order = #NCDHW}> to tensor<1x4x30x30x16xf16, {order = #NCDHW}>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[REORDER1]]
    // CHECK-SAME:      tensor<1x4x30x30x32xf16, {order = #NCDHW}> to tensor<1x4x30x30x16xf16, {order = #NCDHW}>
    // CHECK:       return [[SLICE0]], [[SLICE1]]
    // CHECK-SAME:      tensor<1x4x30x30x16xf16, {order = #NCDHW}>,
    // CHECK-SAME:      tensor<1x4x30x30x16xf16, {order = #NCDHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ReorderWithPermuteCastSubView
module @ReorderWithPermuteCastSubView attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x1x128xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x8x1x128xf16, {order = #NHWC}>) -> (tensor<1x128x3x8xf16, {order = #NHWC}>, tensor<1x8x2x128xf16>) {
    %cst = const.Declare tensor<1x8x2x128xf16> = dense<0.000000e+00> : tensor<1x8x2x128xf32>, [#const.ConvertElemType<f16>]
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x8x1x128xf16, {order = #NHWC}> -> tensor<1x8x1x128xf16>
    %1 = IE.Concat(%cst, %0) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x8x2x128xf16>, tensor<1x8x1x128xf16> -> tensor<1x8x3x128xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x8x3x128xf16> -> tensor<1x128x3x8xf16, {order = #NWHC}>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x128x3x8xf16, {order = #NWHC}> -> tensor<1x128x3x8xf16, {order = #NHWC}>
    %4 = IE.Slice %1 [0, 0, 1, 0] [1, 8, 2, 128] : tensor<1x8x3x128xf16> to tensor<1x8x2x128xf16>
    return %3, %4 : tensor<1x128x3x8xf16, {order = #NHWC}>, tensor<1x8x2x128xf16>
}

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x8x2x128xf16, {order = #NHWC}>
    // CHECK-SAME:      tensor<1x8x2x128xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[CST]], [[ARG0]])
    // CHECK-SAME{LITERAL}    {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} :
    // CHECK-SAME:            tensor<1x8x2x128xf16, {order = #NHWC}>, tensor<1x8x1x128xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x8x3x128xf16, {order = #NHWC}>

    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[CONCAT0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      -> tensor<1x8x3x128xf16>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[REORDER0]]) {dst_order = #NWHC, mem_perm = #NCHW}
    // CHECK-SAME:      -> tensor<1x128x3x8xf16, {order = #NWHC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[PERMUTECAST]]) {dstOrder = #NHWC}
    // CHECK-SAME:      -> tensor<1x128x3x8xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT0]]
    // CHECK-SAME:          [0, 0, 1, 0] [1, 8, 2, 128] : tensor<1x8x3x128xf16, {order = #NHWC}> to tensor<1x8x2x128xf16, {order = #NHWC}>
    // CHECK:       [[REORDER2:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      -> tensor<1x8x2x128xf16>

    // CHECK:       return [[REORDER1]], [[REORDER2]] : tensor<1x128x3x8xf16, {order = #NHWC}>, tensor<1x8x2x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ReorderWithReadValueAndAssign
module @ReorderWithReadValueAndAssign attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x1x128xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x16x3x128xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x2x128xf16> = dense<0.000000e+00> : tensor<1x16x2x128xf32>, [#const.ConvertElemType<f16>]
    %0 = IE.ReadValue(%cst) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    %1 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x16x2x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x3x128xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x16x3x128xf16> -> tensor<1x16x3x128xf16, {order = #NHWC}>
    %4 = IE.Slice %2 [0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16> to tensor<1x16x2x128xf16>
    %5 = IE.Assign(%4) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    return %3 : tensor<1x16x3x128xf16, {order = #NHWC}>
    
    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK-SAME:      tensor<1x16x2x128xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[READVALUE0:%.+]] = IE.ReadValue([[CST]]) {name = "MemoryCellId-2"}
    // CHECK-SAME:      -> tensor<1x16x2x128xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[READVALUE0]], [[ARG0]])
    // CHECK-SAME{LITERAL}    {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} :
    // CHECK-SAME:            tensor<1x16x2x128xf16, {order = #NHWC}>, tensor<1x16x1x128xf16, {order = #NHWC}>
    // CHECK-SAME:            -> tensor<1x16x3x128xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT0]]
    // CHECK-SAME:          [0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16, {order = #NHWC}> to tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK:       [[ASSIGN0:%.+]] = IE.Assign([[SLICE0]]) {name = "MemoryCellId-2"}
    // CHECK-SAME:      -> tensor<1x16x2x128xf16, {order = #NHWC}>
    
    // CHECK:       return [[CONCAT0]] : tensor<1x16x3x128xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ReorderWithReadValueAndAssignNegative
module @ReorderWithReadValueAndAssignNegative attributes {VPU.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x1x128xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x16x3x128xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x2x128xf16> = dense<0.000000e+00> : tensor<1x16x2x128xf32>, [#const.ConvertElemType<f16>]
    %0 = IE.ReadValue(%cst) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    %1 = IE.Negative(%0) : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    %2 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16, {order = #NHWC}>
    %3 = IE.Concat(%2, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x16x2x128xf16, {order = #NHWC}>, tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x3x128xf16, {order = #NHWC}>
    %4 = IE.Slice %3[0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16, {order = #NHWC}> to tensor<1x16x2x128xf16, {order = #NHWC}>
    %5 = IE.Reorder(%4) {dstOrder = #NCHW} : tensor<1x16x2x128xf16, {order = #NHWC}> -> tensor<1x16x2x128xf16>
    %6 = IE.Assign(%5) {name = "MemoryCellId-2"} : tensor<1x16x2x128xf16> -> tensor<1x16x2x128xf16>
    return %3 : tensor<1x16x3x128xf16, {order = #NHWC}>
    
    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x2x128xf16>
    // CHECK-NOT:      tensor<1x16x2x128xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-SAME:     tensor<1x16x2x128xf32>, [#const.ConvertElemType<f16>]
    // CHECK:       [[READVALUE0:%.+]] = IE.ReadValue([[CST]]) {name = "MemoryCellId-2"}
    // CHECK-NOT:      -> tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK:       [[NEGATIVE:%.+]] = IE.Negative([[READVALUE0]]) : tensor<1x16x2x128xf16> 
    // CHECK-SAME:     -> tensor<1x16x2x128xf16>

    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[NEGATIVE]]) {dstOrder = #NHWC}
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[REORDER0]], [[ARG0]])

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT0]]
    // CHECK-SAME:          [0, 0, 1, 0] [1, 16, 2, 128] : tensor<1x16x3x128xf16, {order = #NHWC}> to tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      -> tensor<1x16x2x128xf16>
    // CHECK:       [[ASSIGN0:%.+]] = IE.Assign([[REORDER1]]) {name = "MemoryCellId-2"}
    // CHECK-NOT:      -> tensor<1x16x2x128xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT0]] : tensor<1x16x3x128xf16, {order = #NHWC}>
}

}
