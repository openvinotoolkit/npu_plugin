// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --use-host-pre-post-processing %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

module @ConvertOnly {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf32>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x2x3x4xf32>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16>
func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf16>

    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>

    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf32>

    return %2 : tensor<1x2x3x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK: return [[VAL0]]
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ReorderOnly {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf16>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x2x3x4xf16>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x2x3x4xf16, {order = #NHWC}>) -> tensor<1x2x3x4xf16, {order = #NHWC}>
func @main(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC}>

    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x2x3x4xf16, {order = #NHWC}>

    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x2x3x4xf16>

    return %2 : tensor<1x2x3x4xf16>

    // CHECK: [[VAL0:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK: return [[VAL0]]
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @InputReorderAndConvert {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf32>
    }
    outputsInfo : {
        DataInfo "output" : tensor<24xf32>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x2x3x4xf16, {order = #NHWC}>) -> tensor<24xf16>
func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<24xf32> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf16, {order = #NHWC}>

    %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x2x3x4xf16, {order = #NHWC}>

    %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x2x3x4xf16>
    %4 = IE.Reshape(%3) {shape_value = [24]} : tensor<1x2x3x4xf16> -> tensor<24xf16>

    %5 = IE.Convert(%4) {dstElemType = f32} : tensor<24xf16> -> tensor<24xf32>

    return %5 : tensor<24xf32>

    // CHECK: [[VAL0:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK: [[VAL1:%.+]] = IE.Reorder([[VAL0]]) {dstOrder = #NCHW}
    // CHECK: [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {shape_value = [24]}
    // CHECK: return [[VAL2]]
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @MultipleInputsAndOutputs {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x2x3x4xf32>
        DataInfo "input2" : tensor<1x2x3x4xf32>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x2x3x4xf32>
        DataInfo "output2" : tensor<1x2x3x4xf32>
    }

// CHECK:       func @main([[ARG0:%arg[0-9]+]]: tensor<1x2x3x4xf16, {order = #NHWC}>,
// CHECK-SAME:             [[ARG1:%arg[0-9]+]]: tensor<1x2x3x4xf16, {order = #NHWC}>)
// CHECK-SAME:      -> (tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16>)
func @main(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf16>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC}>

    %2 = IE.Reorder(%arg1) {dstOrder = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    %3 = IE.Convert(%2) {dstElemType = f16} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf16, {order = #NHWC}>

    %4 = IE.Add(%1, %3) { auto_broadcast = "NUMPY" } :
        tensor<1x2x3x4xf16, {order = #NHWC}>, tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x2x3x4xf16>
    %5 = IE.Multiply(%1, %3) { auto_broadcast = "NUMPY" } :
        tensor<1x2x3x4xf16, {order = #NHWC}>, tensor<1x2x3x4xf16, {order = #NHWC}> -> tensor<1x2x3x4xf16>

    %6 = IE.Convert(%4) {dstElemType = f32} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf32>
    %7 = IE.Convert(%5) {dstElemType = f32} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf32>

    return %6, %7: tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.Add([[ARG0]], [[ARG1]])
    // CHECK: [[VAL1:%.+]] = IE.Multiply([[ARG0]], [[ARG1]])
    // CHECK: return [[VAL0]], [[VAL1]]
}

}
