// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --set-internal-memory-space="memory-space=DDR" %s | FileCheck %s

// CHECK-LABEL: @Test
module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf16>
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>

    %1 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>

    IERT.SoftMax(%1, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>

    return

    // CHECK: [[VAR0:%[0-9]*]] = alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[ARG0]], [[VAR0]])

    // CHECK: [[VAR1:%[0-9]*]] = alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]])

    // CHECK: IERT.SoftMax([[VAR1]], [[ARG1]])

    // CHECK: return
}

}
