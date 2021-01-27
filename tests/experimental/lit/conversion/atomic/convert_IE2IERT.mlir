// RUN: vpux-opt --split-input-file --convert-IE-to-IERT %s | FileCheck %s

//
// The 'bufferize-IE' pass:
//
//   * Updates only Function inner regions.
//   * Doesn't touch `IE.CNNNetwork` Operation.
//   * Doesn't change Function signatures.
//   * Doesn't update `std.constant` Operations.
//   * Replaces only Layer Operations.
//

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf32>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf32>
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

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x2x2x2xf32>
    }

// CHECK: func @main() -> tensor<1x2x2x2xf16> {
func @main() -> tensor<1x2x2x2xf16> {
    %0 = IE.Constant tensor<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK: [[VAR0:%[0-9]*]] = IERT.Constant memref<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf32>
    // CHECK: [[VAR1:%[0-9]*]] = tensor_load [[VAR0]]
    // CHECK: return [[VAR1]] : tensor<1x2x2x2xf16>
}

}
