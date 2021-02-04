// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --lower-IERT-to-VPUIP %s | FileCheck %s

//
// The 'lower-IERT-to-VPUIP' pass:
//
//   * Replaces Layer Operations with VPUIP Tasks.
//   * Adds `VPUIP.Graph` Operation.
//   * Removes `std.global_memref` Operations.
//

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedExecutors
// CHECK:           IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:           IERT.ExecutorResource 1 of "NCE_Cluster"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf32>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf32>
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1x1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1x1x1000xf16>) {
func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) {
    %0 = IERT.StaticAlloc<0> -> memref<1x1x1x1000xf16, "DDR">
    IERT.SoftMax(%arg0, %0) {axisInd = 3 : i32} : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16, "DDR">
    linalg.copy(%0, %arg1) : memref<1x1x1x1000xf16, "DDR">, memref<1x1x1x1000xf16>
    return

    // CHECK:       [[VAR0:%[0-9]*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1000xf16, "DDR">)

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1x1x1000xf16, "DDR">)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1x1x1000xf16>)
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedExecutors
// CHECK:           IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:           IERT.ExecutorResource 1 of "NCE_Cluster"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x2x2x2xf32>
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x2x2x2xf16>) {
func @main(%arg0: memref<1x2x2x2xf16>) {
    %0 = IERT.Constant memref<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf32>
    linalg.copy(%0, %arg0) : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
    return

    // CHECK:       [[VAR0:%[0-9]*]] = VPUIP.DeclareConstantTensorOp memref<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf32>
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs([[ARG0]] : memref<1x2x2x2xf16>)
}

}
