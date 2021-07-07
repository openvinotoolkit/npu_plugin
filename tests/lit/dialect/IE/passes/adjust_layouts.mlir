// RUN: vpux-opt --split-input-file --adjust-layouts %s | FileCheck %s

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @InOutNHCW
module @InOutNHCW attributes {VPUIP.arch = "VPU3700", VPUIP.compilationMode = "ReferenceSW"} {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : tensor<1x8x4x2xf16, {order = #NHCW}>
    }
    outputsInfo : {
        IE.DataInfo "prob" : tensor<1x8x4x2xf16, {order = #NHCW}>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16, {order = #NHCW}>) -> tensor<1x8x4x2xf16, {order = #NHCW}> {
func @main(%arg0: tensor<1x8x4x2xf16, {order = #NHCW}>) -> tensor<1x8x4x2xf16, {order = #NHCW}> {
    %0 = IE.GRN(%arg0) {bias = 1.0 : f32} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i32} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>
    %2 = IE.GRN(%1) {bias = 1.0 : f32} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>
    return %2 : tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR1:%.+]] = IE.GRN([[VAR0]]) {bias = 1.000000e+00 : f32} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NHCW} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: [[VAR3:%.+]] = IE.SoftMax([[VAR2]]) {axisInd = 1 : i32} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: [[VAR4:%.+]] = IE.Reorder([[VAR3]]) {dstOrder = #NCHW} : tensor<1x8x4x2xf16, {order = #NHCW}> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR5:%.+]] = IE.GRN([[VAR4]]) {bias = 1.000000e+00 : f32} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR6:%.+]] = IE.Reorder([[VAR5]]) {dstOrder = #NHCW} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHCW}>

    // CHECK: return [[VAR6]] : tensor<1x8x4x2xf16, {order = #NHCW}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DifferentOrders
module @DifferentOrders attributes {VPUIP.arch = "VPU3700", VPUIP.compilationMode = "ReferenceSW"} {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : tensor<1x8x4x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : tensor<1x8x4x2xf16, {order = #NHWC}>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x8x4x2xf16>) -> tensor<1x8x4x2xf16, {order = #NHWC}> {
func @main(%arg0: tensor<1x8x4x2xf16>) -> tensor<1x8x4x2xf16, {order = #NHWC}> {
    %0 = IE.GRN(%arg0) {bias = 1.0 : f32} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    return %0 : tensor<1x8x4x2xf16, {order = #NHWC}>

    // CHECK: [[VAR0:%.+]] = IE.GRN([[ARG0]]) {bias = 1.000000e+00 : f32} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16>
    // CHECK: [[VAR1:%.+]] = IE.Reorder([[VAR0]]) {dstOrder = #NHWC} : tensor<1x8x4x2xf16> -> tensor<1x8x4x2xf16, {order = #NHWC}>
    // CHECK: return [[VAR1]] : tensor<1x8x4x2xf16, {order = #NHWC}>
}

}

// -----

// CHECK-LABEL: @HwOp
module @HwOp attributes {VPUIP.arch = "VPU3700", VPUIP.compilationMode = "ReferenceHW"} {

IERT.RunTimeResources
    availableMemory :  {
        IERT.MemoryResource 201326592 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 917504 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory : {
    }
    executors : {
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : tensor<1x16x30x30xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : tensor<1x16x15x13xf16>
    }

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x30x30xf16>) -> tensor<1x16x15x13xf16> {
func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x16x15x13xf16> {
   %0 = IE.MaxPool(%arg0) {
        kernel_size = [5 : i32, 5 : i32],
        pads_begin = [2 : i32, 0 : i32],
        pads_end = [2 : i32, 0 : i32],
        rounding_type = "FLOOR",
        strides = [2 : i32, 2 : i32]
    } : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>
    return %0 : tensor<1x16x15x13xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.MaxPool([[VAR0]])
    // CHECK-SAME:      tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x15x13xf16, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW} : tensor<1x16x15x13xf16, {order = #NHWC}> -> tensor<1x16x15x13xf16>
    // CHECK:       return [[VAR2]] : tensor<1x16x15x13xf16>
}

}
