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

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    linalg.copy(%0, %arg1) : memref<1x1000xf16>, memref<1x1000xf16>
    dealloc %0 : memref<1x1000xf16>
    return

    // CHECK:       [[VAR0:%[0-9]*]] = alloc() : memref<1x1000xf16>
    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1000xf16>)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)
    // CHECK-NEXT:  dealloc [[VAR0]] : memref<1x1000xf16>
    // CHECK-NEXT:  return
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

global_memref "private" constant @cst : memref<1x2x2x2xf16> = dense<1.0>

// CHECK-NOT: global_memref

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x2x2x2xf16>) {
func @main(%arg0: memref<1x2x2x2xf16>) {
    %0 = get_global_memref @cst : memref<1x2x2x2xf16>
    linalg.copy(%0, %arg0) : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
    return

    // CHECK:       [[VAR0:%[0-9]*]] = VPUIP.DeclareConstantTensorOp
    // CHECK-SAME:      dense<1.000000e+00>
    // CHECK-SAME:      -> memref<1x2x2x2xf16>
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs([[ARG0]] : memref<1x2x2x2xf16>)
    // CHECK-NEXT:  return
}

}
