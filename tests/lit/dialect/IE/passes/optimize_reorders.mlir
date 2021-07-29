// RUN: vpux-opt --split-input-file --optimize-reorders %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReorderWithSubView
module @ReorderWithSubView attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"} {

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16>)
func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x4x4x2xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    %1 = tensor.extract_slice %0[0, 2, 0, 0] [1, 4, 4, 2] [1, 1, 1, 1] : tensor<1x8x4x2xf16, {order = #NHWC}> to tensor<1x4x4x2xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x4x4x2xf16, {order = #NHWC}> -> tensor<1x4x4x2xf16>
    return %2 : tensor<1x4x4x2xf16>

    // CHECK:       [[VAR0:%.+]] = tensor.extract_slice [[ARG0]]
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
        pads_begin_attr = [0, 0, 0, 0],
        pads_end_attr = [0, 13, 0, 0]
    } : tensor<1x3x30x30xf16> -> tensor<1x16x30x30xf16>

    %2 = IE.MaxPool(%1) {
        kernel_size = [5, 5],
        pads_begin = [2, 0],
        pads_end = [2, 0],
        rounding_type = "FLOOR",
        strides = [2, 2]
    } : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>

    %3 = tensor.extract_slice %2[0, 0, 0, 0] [1, 3, 15, 13] [1, 1, 1, 1] : tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

    %4 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x3x15x13xf16> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    return %4 : tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK:       [[VAR0:%.+]] = IE.Expand([[ARG0]]
    // CHECK-SAME:      tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NCHW}
    // CHECK-SAME:      tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16>

    // CHECK:       [[VAR2:%.+]] = IE.MaxPool([[VAR1]])
    // CHECK-SAME:      tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>

    // CHECK:       [[VAR3:%.+]] = tensor.extract_slice [[VAR2]]
    // CHECK-SAME:      tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

    // CHECK:       [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NHWC}
    // CHECK-SAME:      tensor<1x3x15x13xf16> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    // CHECK        return %[[VAR4]] : tensor<1x3x15x13xf16, {order = #NHWC}>
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
