// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --static-allocation="memory-space=DDR" %s | FileCheck %s

// CHECK-LABEL: @LinearGraph
module @LinearGraph {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf16>
    }

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           IERT.MemoryResource 4096 bytes of "DDR"

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16, "DDR">

    %1 = alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">
    dealloc %0 : memref<1x1000xf16, "DDR">

    %2 = alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%1, %2) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">
    dealloc %1 : memref<1x1000xf16, "DDR">

    linalg.copy(%2, %arg1) : memref<1x1000xf16, "DDR">, memref<1x1000xf16>
    dealloc %2 : memref<1x1000xf16, "DDR">

    return

    // CHECK-NOT:   alloc
    // CHECK:       [[VAR0:%[0-9]*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, "DDR">

    // CHECK:       IERT.SoftMax([[ARG0]], [[VAR0]])

    // CHECK-NOT:   alloc
    // CHECK:       [[VAR1:%[0-9]*]] = IERT.StaticAlloc<2048> -> memref<1x1000xf16, "DDR">

    // CHECK:       IERT.SoftMax([[VAR0]], [[VAR1]])
    // CHECK-NOT:   dealloc

    // CHECK-NOT:   alloc
    // CHECK:       [[VAR2:%[0-9]*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, "DDR">

    // CHECK:       IERT.SoftMax([[VAR1]], [[VAR2]])
    // CHECK-NOT:   dealloc

    // CHECK:       linalg.copy([[VAR2]], [[ARG1]])
    // CHECK-NOT:   dealloc

    // CHECK-NEXT:  return
}

}
