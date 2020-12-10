// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --lower-IE-to-IERT %s | FileCheck %s

//
// The 'lower-IE-to-IERT' pass:
//
//   * Fully replaces IE Dialect with IERT Dielect.
//   * Changes all Values types from `tensor` to `memref`.
//   * Changes Function results tensors to arguments.
//   * Replaces `std.constant` Operations with `global_memref`/`get_global_memref`.
//   * Inserts `linalg.copy` for `std.constant` as result case.
//   * Inserts `std.dealloc` Operations for inner buffers.
//

// CHECK: #NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: IERT.CNNNetwork "SingleLayer" at @main

IE.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        // CHECK: IERT.DataInfo "data", f32, #NC
        IE.DataInfo "data", f32, "NC"
    }
    outputsInfo : {
        // CHECK: IERT.DataInfo "prob", f32, #NC
        IE.DataInfo "prob", f32, "NC"
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: IERT.SoftMax([[ARG0]], [[ARG1]]) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    // CHECK: return
}

// -----

// CHECK: #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: IERT.CNNNetwork "ConstantLayer" at @main

IE.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: IERT.DataInfo "output", f32, #NCHW
        IE.DataInfo "output", f32, "NCHW"
    }

// CHECK: global_memref "private" constant [[CST0:@[a-z0-9_]+]] : memref<1x2x2x2xf16>

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x2x2x2xf16>) {
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

    // CHECK: [[VAR0:%[0-9]*]] = get_global_memref [[CST0]] : memref<1x2x2x2xf16>
    // CHECK: linalg.copy([[VAR0]], [[ARG0]]) : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
    // CHECK: return
}
