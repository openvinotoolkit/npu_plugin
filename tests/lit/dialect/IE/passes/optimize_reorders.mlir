// RUN: vpux-opt --split-input-file --optimize-reorders %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubView
module @ReorderWithSubView attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

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

// CHECK-LABEL: @ReorderWithExpand
module @ReorderWithExpand attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

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

    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16>

    // CHECK:       [[VAR2:%.+]] = IE.MaxPool([[VAR1]])
    // CHECK-SAME:      tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>

    // CHECK:       [[VAR3:%.+]] = IE.Slice [[VAR2]]
    // CHECK-SAME:      tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

    // CHECK:       [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x3x15x13xf16> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK        return %[[VAR4]] : tensor<1x3x15x13xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127, 0.0013264333169291339:127,9.8713551919291337E-4:127}>
!qElemType3 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>
!qElemType4 = type !quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127, 0.0013264333169291339:127,9.8713551919291337E-4:127,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>

module @ReorderWithQuantExpandAndSlice attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceHW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x3x30x30x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>>)
func @main(%arg0: tensor<1x3x30x30x!qElemType1>) -> tensor<1x3x15x13x!qElemType2> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x30x30x!qElemType1> -> tensor<1x3x30x30x!qElemType1, {order = #NHWC}>

    %1 = IE.Expand(%0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x30x30x!qElemType1, {order = #NHWC}> -> tensor<1x16x30x30x!qElemType3, {order = #NHWC}>

    %2 = IE.MaxPool(%1) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = "FLOOR",
        strides = [2, 2]
    } : tensor<1x16x30x30x!qElemType3, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType4, {order = #NHWC}>

    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x16x15x13x!qElemType4, {order = #NHWC}> -> tensor<1x16x15x13x!qElemType4>

    %4 = IE.Slice %3 [0, 0, 0, 0] [1, 3, 15, 13] : tensor<1x16x15x13x!qElemType4> to tensor<1x3x15x13x!qElemType2>

    return %4 : tensor<1x3x15x13x!qElemType2>

    // CHECK: [[VAR0:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
    // CHECK-SAME:     tensor<1x3x30x30x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>> ->
    // CHECK-SAME:     tensor<1x16x30x30x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>>

    // CHECK: [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} :
    // CHECK-SAME:     tensor<1x16x30x30x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>> ->
    // CHECK-SAME:     tensor<1x16x30x30x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>, {order = #NHWC}>

    // CHECK: [[VAR2:%.+]] = IE.MaxPool([[VAR1]])

    // CHECK: [[VAR3:%.+]] = IE.Slice [[VAR2]] [0, 0, 0, 0] [1, 3, 15, 13] :
    // CHECK-SAME:     tensor<1x16x15x13x!quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127,0.0013264333169291339:127,9.8713551919291337E-4:127,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>, {order = #NHWC}> to
    // CHECK-SAME:     tensor<1x3x15x13x!quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127,0.0013264333169291339:127,9.8713551919291337E-4:127}>, {order = #NHWC}>

    // CHECK: [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NCHW} :
    // CHECK-SAME:     tensor<1x3x15x13x!quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127,0.0013264333169291339:127,9.8713551919291337E-4:127}>, {order = #NHWC}> ->
    // CHECK-SAME:     tensor<1x3x15x13x!quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127,0.0013264333169291339:127,9.8713551919291337E-4:127}>>

    // CHECK: return [[VAR4]] : tensor<1x3x15x13x!quant.uniform<u8<0:254>:f16:1, {5.0750492125984249E-4:127,0.0013264333169291339:127,9.8713551919291337E-4:127}>>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSplit
module @ReorderWithSplit attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

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

// CHECK-LABEL: @ReorderWithConcat
module @ReorderWithConcat attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<1x1x30x30xf16, {order = #NHWC}>)
func @main(%arg0: tensor<1x1x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x1x30x30xf16, {order = #NHWC}>)
        -> tensor<1x2x30x30xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x1x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16>
    %2 = IE.Concat(%0, %1) {axis = 1} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x2x30x30xf16>
    %3 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x2x30x30xf16> -> tensor<1x2x30x30xf16, {order = #NHWC}>
    return %3 : tensor<1x2x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG0]], [[ARG1]]) {axis = 1 : i64}
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x30x30xf16, {order = #NHWC}>
    // CHECK:       return [[VAR0]] : tensor<1x2x30x30xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithExpandTwoBranches
module @ReorderWithExpandTwoBranches attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

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
