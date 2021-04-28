// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --static-allocation="memory-space=DDR" %s | FileCheck %s

//
// The 'static-allocation' pass:
//
//   * Starts from `IE.CNNNetwork` entry point Function.
//   * It replaces all `memref.alloc`/`memref.dealloc` Operations, that works with specific memory space,
//     with single `IERT.StaticAlloc` Operation.
//

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

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16, "DDR">
    %1 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">

    %2 = memref.alloc() : memref<1x1000xf16, "DDR">
    %3 = IERT.SoftMax {axisInd = 1 : i32} inputs(%1: memref<1x1000xf16, "DDR">) outputs(%2 : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">
    memref.dealloc %0 : memref<1x1000xf16, "DDR">

    %4 = memref.alloc() : memref<1x1000xf16, "DDR">
    %5 = IERT.SoftMax {axisInd = 1 : i32} inputs(%3: memref<1x1000xf16, "DDR">) outputs(%4 : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">
    memref.dealloc %2 : memref<1x1000xf16, "DDR">

    %6 = IERT.Copy inputs(%5 : memref<1x1000xf16, "DDR">) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    memref.dealloc %4 : memref<1x1000xf16, "DDR">

    return %6 : memref<1x1000xf16>

    // CHECK-NOT:   memref.alloc
    // CHECK:       [[VAR0:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, "DDR">

    // CHECK:       [[VAR1:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1000xf16, "DDR">)

    // CHECK-NOT:   memref.alloc
    // CHECK:       [[VAR2:%.*]] = IERT.StaticAlloc<2048> -> memref<1x1000xf16, "DDR">

    // CHECK:       [[VAR3:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x1000xf16, "DDR">)
    // CHECK-NOT:   memref.dealloc

    // CHECK-NOT:   memref.alloc
    // CHECK:       [[VAR4:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, "DDR">

    // CHECK:       [[VAR5:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x1000xf16, "DDR">)
    // CHECK-NOT:   memref.dealloc

    // CHECK:       [[VAR6:%.*]] = IERT.Copy
    // CHECK-NOT:   memref.dealloc

    // CHECK:       return [[VAR6]] : memref<1x1000xf16>
}

}
