// RUN: vpux-opt --split-input-file --add-buffers-for-net-results %s | FileCheck %s

// CHECK: func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @SingleLayer(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) -> memref<1x1000xf16>
    return %1 : memref<1x1000xf16>

    // CHECK: [[VAR1:%.*]] = IERT.Copy inputs(%1 : memref<1x1000xf16>) outputs([[ARG1]] : memref<1x1000xf16>) -> memref<1x1000xf16>
    // CHECK: return [[VAR1]] : memref<1x1000xf16>
}
