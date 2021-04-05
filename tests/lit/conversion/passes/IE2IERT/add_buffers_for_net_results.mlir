// RUN: vpux-opt --split-input-file --add-buffers-for-net-results %s | FileCheck %s

// CHECK: func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @SingleLayer(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    return %0 : memref<1x1000xf16>

    // CHECK: linalg.copy(%0, [[ARG1]]) : memref<1x1000xf16>, memref<1x1000xf16>
    // CHECK: return [[ARG1]] : memref<1x1000xf16>
}
