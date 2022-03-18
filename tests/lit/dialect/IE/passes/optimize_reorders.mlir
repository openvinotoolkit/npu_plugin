// RUN: vpux-opt --split-input-file --optimize-reorders %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubView
module @ReorderWithSubView attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

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
module @ReorderWithTwoUsersSubView attributes {VPU.arch = "KMB", VPU.compilationMode = "DefaultHW"} {

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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpand
module @ReorderWithExpand attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

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

module @ReorderWithQuantExpandAndSlice attributes {VPU.arch = "KMB", VPU.compilationMode = "DefaultHW"} {

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
module @ReorderWithSplit attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

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
module @ReorderWithSplitMultipleUses attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

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
module @ReorderWithConcat attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

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
module @ReorderWithConcatWithConsts attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = const.Declare tensor<1x1x30x30xf16> = #const.Content<dense<1.0> : tensor<1x1x30x30xf16>>
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
module @ReorderPropagationWithConcat attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

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
module @ReorderWithExpandTwoBranches attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

// CHECK:       func @main([[ARG0:%arg[0-9]+]]: tensor<1x24x56x56xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x24x56x56xf16, {order = #NHWC}>) -> tensor<1x32x56x56xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]>
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
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]>

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
module @ReorderWithLayer attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"} {

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
module @ReorderWithQuantizeCast attributes {VPU.arch = "KMB", VPU.compilationMode = "DefaultHW"} {

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

module @ReorderWithQuantizeCastTwoBranches attributes {VPU.arch = "KMB", VPU.compilationMode = "DefaultHW"} {

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
module @CMajorToZMajorConv attributes {VPU.arch = "KMB", VPU.compilationMode = "DefaultHW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
func @main(%arg0: tensor<1x3x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}> =
        #const.Content<dense<1.0> : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>]>

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
