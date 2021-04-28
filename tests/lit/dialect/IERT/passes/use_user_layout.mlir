// RUN: vpux-opt --split-input-file --use-user-layout %s | FileCheck %s

//
// The 'use-user-layout' pass:
//
//   * Adds user layouts into function signature
//   * Inserts Reorder operation for input/output if needed
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @InOutNHWC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x8x4x2xf16, #NHWC>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x8x4x2xf16, #NHWC>
    }

// CHECK: func @main([[ARG0:%.*]]: memref<1x8x4x2xf16, #NHWC>, [[ARG1:%.*]]: memref<1x8x4x2xf16, #NHWC>) -> memref<1x8x4x2xf16, #NHWC> {
func @main(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %0 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.Reorder inputs([[ARG0]] : memref<1x8x4x2xf16, #NHWC>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR3:%.*]] = IERT.SoftMax {axisInd = 1 : i32} inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs([[VAR2]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR4:%.*]] = IERT.Reorder inputs([[VAR3]] : memref<1x8x4x2xf16>) outputs([[ARG1]] : memref<1x8x4x2xf16, #NHWC>) -> memref<1x8x4x2xf16, #NHWC>
    // CHECK: return [[VAR4]] : memref<1x8x4x2xf16, #NHWC>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @InNHWC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x8x4x2xf16, #NHWC>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x8x4x2xf16>
    }

// CHECK: func @main([[ARG0:%.*]]: memref<1x8x4x2xf16, #NHWC>, [[ARG1:%.*]]: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
func @main(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %0 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.Reorder inputs([[ARG0]] : memref<1x8x4x2xf16, #NHWC>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.SoftMax {axisInd = 1 : i32} inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs([[ARG1]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: return [[VAR2]] : memref<1x8x4x2xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @TwoOutputsNHWC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x8x4x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "output1" : memref<1x8x4x2xf16, #NHWC>
        IE.DataInfo "output2" : memref<1x20x8x4xf16, #NHWC>
    }

// CHECK: func @main([[ARG0:%.*]]: memref<1x8x4x2xf16>, [[ARG1:%.*]]: memref<1x8x4x2xf16, #NHWC>, [[ARG2:%.*]]: memref<1x20x8x4xf16, #NHWC>) -> (memref<1x8x4x2xf16, #NHWC>, memref<1x20x8x4xf16, #NHWC>) {
func @main(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>, %arg2: memref<1x20x8x4xf16>) -> (memref<1x8x4x2xf16>, memref<1x20x8x4xf16>) {
    %0 = IERT.Constant memref<1x4x8x20xf16> = dense<2.000000e+00> : tensor<1x4x8x20xf16>
    %1 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %2 = IERT.GenericReshape inputs(%arg2 : memref<1x20x8x4xf16>) -> memref<1x4x8x20xf16>
    %3 = IERT.SoftMax {axisInd = 1 : i32} inputs(%0 : memref<1x4x8x20xf16>) outputs(%2 : memref<1x4x8x20xf16>) -> memref<1x4x8x20xf16>
    %4 = IERT.GenericReshape inputs(%3 : memref<1x4x8x20xf16>) -> memref<1x20x8x4xf16>

    return %1, %4 : memref<1x8x4x2xf16>, memref<1x20x8x4xf16>

    // CHECK: [[VAR0:%.*]] = IERT.Constant memref<1x4x8x20xf16> = dense<2.000000e+00> : tensor<1x4x8x20xf16>
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.SoftMax {axisInd = 1 : i32} inputs([[ARG0]] : memref<1x8x4x2xf16>) outputs([[VAR1]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    // CHECK: [[VAR3:%.*]] = memref.alloc() : memref<1x20x8x4xf16>
    // CHECK: [[VAR4:%.*]] = IERT.GenericReshape inputs([[VAR3]] : memref<1x20x8x4xf16>) -> memref<1x4x8x20xf16>
    // CHECK: [[VAR5:%.*]] = IERT.SoftMax {axisInd = 1 : i32} inputs([[VAR0]] : memref<1x4x8x20xf16>) outputs([[VAR4]] : memref<1x4x8x20xf16>) -> memref<1x4x8x20xf16>
    // CHECK: [[VAR6:%.*]] = IERT.GenericReshape inputs([[VAR5]] : memref<1x4x8x20xf16>) -> memref<1x20x8x4xf16>
    // CHECK: [[VAR7:%.*]] = IERT.Reorder inputs([[VAR2]] : memref<1x8x4x2xf16>) outputs([[ARG1]] : memref<1x8x4x2xf16, #NHWC>) -> memref<1x8x4x2xf16, #NHWC>
    // CHECK: [[VAR8:%.*]] = IERT.Reorder inputs([[VAR6]] : memref<1x20x8x4xf16>) outputs([[ARG2]] : memref<1x20x8x4xf16, #NHWC>) -> memref<1x20x8x4xf16, #NHWC>

    // CHECK: return [[VAR7]], [[VAR8]] : memref<1x8x4x2xf16, #NHWC>, memref<1x20x8x4xf16, #NHWC>
}

}
