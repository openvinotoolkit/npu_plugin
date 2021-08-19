// RUN: vpux-opt --split-input-file --use-user-layout %s | FileCheck %s

//
// The 'use-user-layout' pass:
//
//   * Adds user layouts into function signature
//   * Inserts Reorder operation for input/output if needed
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InOutNHWC
module @InOutNHWC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : tensor<1x8x4x2xf16, {order = #NHWC}>
    }
    outputsInfo : {
        IE.DataInfo "prob" : tensor<1x8x4x2xf16, {order = #NHWC}>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16, {order = #NHWC}>) -> tensor<1x8x4x2xf16, {order = #NHWC}> {
func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x8x4x2xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    return %0 : tensor<1x8x4x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHWC}> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR1:%.+]] = IE.SoftMax([[VAR0]]) {axisInd = 1 : i64} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    // CHECK: return [[VAR2]] : tensor<1x8x4x2xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InNHWC
module @InNHWC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : tensor<1x8x4x2xf16, {order = #NHWC}>
    }
    outputsInfo : {
        IE.DataInfo "prob" : tensor<1x8x4x2xf16>
    }

// CHECK:       func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16, {order = #NHWC}>)
// CHECK-SAME:      -> tensor<1x8x4x2xf16> {
func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x8x4x2xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    return %0 : tensor<1x8x4x2xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NCHW} :
    // CHECK-SAME:      tensor<1x8x4x2xf16, {order = #NHWC}> -> tensor<1x8x4x2xf16>
    // CHECK:       [[VAR1:%.+]] = IE.SoftMax([[VAR0]]) {axisInd = 1 : i64} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK:       return [[VAR1]] : tensor<1x8x4x2xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoOutputsNHWC
module @TwoOutputsNHWC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data1" : tensor<1x8x4x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "output1" : tensor<1x8x4x2xf16, {order = #NHWC}>
        IE.DataInfo "output2" : tensor<1x20x8x4xf16, {order = #NHWC}>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16>) -> (tensor<1x8x4x2xf16, {order = #NHWC}>, tensor<1x20x8x4xf16, {order = #NHWC}>) {
func @main(%arg0: tensor<1x8x4x2xf16>) -> (tensor<1x8x4x2xf16>, tensor<1x20x8x4xf16>) {
    %0 = const.Declare tensor<1x4x8x20xf16> = #const.Content<dense<2.000000e+00> : tensor<1x4x8x20xf16>>
    %1 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    %2 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x4x8x20xf16> -> tensor<1x4x8x20xf16>
    %3 = IE.Reshape(%2) { shape_value = [1, 20, 8, 4] } : tensor<1x4x8x20xf16> -> tensor<1x20x8x4xf16>
    return %1, %3 : tensor<1x8x4x2xf16>, tensor<1x20x8x4xf16>

    // CHECK: [[VAR0:%.+]] = const.Declare tensor<1x4x8x20xf16> = #const.Content<dense<2.000000e+00> : tensor<1x4x8x20xf16>>
    // CHECK: [[VAR1:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR2:%.+]] = IE.SoftMax([[VAR0]]) {axisInd = 1 : i64} : tensor<1x4x8x20xf16> -> tensor<1x4x8x20xf16>
    // CHECK: [[VAR3:%.+]] = IE.Reshape([[VAR2]]) {shape_value = [1, 20, 8, 4]} : tensor<1x4x8x20xf16> -> tensor<1x20x8x4xf16>
    // CHECK: [[VAR4:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    // CHECK: [[VAR5:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NHWC} : tensor<1x20x8x4xf16> -> tensor<1x20x8x4xf16, {order = #NHWC}>
    // CHECK: return [[VAR4]], [[VAR5]] : tensor<1x8x4x2xf16, {order = #NHWC}>, tensor<1x20x8x4xf16, {order = #NHWC}>
}

}
