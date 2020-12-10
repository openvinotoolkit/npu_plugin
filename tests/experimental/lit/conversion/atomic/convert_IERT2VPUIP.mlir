// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --convert-IERT-to-VPUIP %s | FileCheck %s

//
// The 'convert-IERT-to-VPUIP' pass:
//
//   * Updates only Function inner regions.
//   * Doesn't touch `IERT.CNNNetwork` Operation.
//   * Doesn't remove `std.global_memref` Operations.
//   * Replaces only Layer Operations.
//

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: IERT.CNNNetwork "SingleLayer" at @main

IERT.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        IERT.DataInfo "data", f32, #NC
    }
    outputsInfo : {
        IERT.DataInfo "prob", f32, #NC
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    IERT.SoftMax(%arg0, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    return

    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      maxShaves = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)
    // CHECK-NEXT:  return
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: IERT.CNNNetwork "ConstantLayer" at @main

IERT.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo :  {
        IERT.DataInfo "output", f32, #NCHW
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
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs([[ARG0]] : memref<1x2x2x2xf16>)
    // CHECK-NEXT:  return
}
