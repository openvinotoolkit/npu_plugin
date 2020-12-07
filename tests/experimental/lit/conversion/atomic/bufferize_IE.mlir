// RUN: vpux-opt -split-input-file -bufferize-IE %s | FileCheck %s

//
// The 'bufferize-IE' pass:
//
//   * Updates only Function inner regions.
//   * Doesn't touch `IE.CNNNetwork` Operation.
//   * Doesn't change Function signatures.
//   * Doesn't update `std.constant` Operations.
//   * Replaces only Layer Operations.
//

// CHECK-LABEL: IE.CNNNetwork "SingleLayer" at @main

IE.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        IE.DataInfo "data", f32, "NC"
    }
    outputsInfo : {
        IE.DataInfo "prob", f32, "NC"
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%[0-9]*]] = tensor_to_memref [[ARG0]] : memref<1x1000xf16>
    // CHECK: [[VAR1:%[0-9]*]] = alloc() : memref<1x1000xf16>
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]]) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    // CHECK: [[VAR2:%[0-9]*]] = tensor_load [[VAR1]] : memref<1x1000xf16>
    // CHECK: return [[VAR2]] : tensor<1x1000xf16>
}

// -----

// CHECK-LABEL: IE.CNNNetwork "ConstantLayer" at @main

IE.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "output", f32, "NCHW"
    }

// CHECK: func @main() -> tensor<1x2x2x2xf16> {
func @main() -> tensor<1x2x2x2xf16> {
    %0 = constant
        dense<[
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0]
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0]
                ]
            ]
        ]> : tensor<1x2x2x2xf16>

    return %0 : tensor<1x2x2x2xf16>

    // CHECK:       [[CST0:%cst[0-9]*]] = constant
    // CHECK-SAME:      tensor<1x2x2x2xf16>
    // CHECK:       return [[CST0]] : tensor<1x2x2x2xf16>
}
