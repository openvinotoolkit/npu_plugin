// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --convert-IERT-to-VPUIP %s | FileCheck %s

//
// The 'convert-IERT-to-VPUIP' pass:
//
//   * Updates only Function inner regions.
//   * Doesn't touch `IERT.CNNNetwork` Operation.
//   * Doesn't remove `std.global_memref` Operations.
//   * Replaces only Layer Operations and `IERT.StaticAlloc`.
//

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf32>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf32>
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    IERT.SoftMax(%arg0, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    return

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo :  {
        IE.DataInfo "output" : memref<1x2x2x2xf32>
    }

// CHECK: global_memref
global_memref "private" constant @cst : memref<1x2x2x2xf16> = dense<1.0>

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x2x2x2xf16>) {
func @main(%arg0: memref<1x2x2x2xf16>) {
    %0 = get_global_memref @cst : memref<1x2x2x2xf16>
    linalg.copy(%0, %arg0) : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
    return

    // CHECK:       [[VAR0:%[0-9]*]] = VPUIP.DeclareConstantTensorOp
    // CHECK-SAME:      dense<1.000000e+00>
    // CHECK-SAME:      -> memref<1x2x2x2xf16>

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs([[ARG0]] : memref<1x2x2x2xf16>)
}

}

// -----

// CHECK-LABEL: @StaticAlloc
module @StaticAlloc {

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf32>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf32>
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = IERT.StaticAlloc<0> -> memref<1x1000xf16, "DDR">
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16, "DDR">

    %1 = IERT.StaticAlloc<2048> -> memref<1x1000xf16, "DDR">
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">

    IERT.SoftMax(%1, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16>

    return

    // CHECK:       [[VAR0:%[0-9]*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1000xf16, "DDR">

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1000xf16, "DDR">)

    // CHECK:       [[VAR1:%[0-9]*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<1x1000xf16, "DDR">

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x1000xf16, "DDR">)

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)
}

}
