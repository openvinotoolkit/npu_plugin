// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --lower-IERT-to-VPUIP %s | FileCheck %s

//
// The 'lower-IERT-to-VPUIP' pass:
//
//   * Replaces `IERT.CNNNetwork` Operation with `VPUIP.Graph`.
//   * Removes `std.global_memref` Operations.
//   * Replaces Layer Operations.
//

#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK:       VPUIP.Graph "SingleLayer" at @main
// CHECK-SAME:      options : "NONE"
IERT.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        // CHECK: VPUIP.TensorInfo "data", f32, #NC
        IERT.DataInfo "data", f32, #NC
    }
    outputsInfo : {
        // CHECK: VPUIP.TensorInfo "prob", f32, #NC
        IERT.DataInfo "prob", f32, #NC
    }

// CHECK-NOT: IERT.CNNNetwork

// CHECK:   IERT.RunTimeResources
// CHECK:       usedExecutors
// CHECK:           IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:           IERT.ExecutorResource 1 of "NCE_Cluster"

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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

// CHECK:       VPUIP.Graph "ConstantLayer" at @main
// CHECK-SAME:      options : "NONE"
IERT.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: VPUIP.TensorInfo "output", f32, #NCHW
        IERT.DataInfo "output", f32, #NCHW
    }

// CHECK-NOT: IERT.CNNNetwork

// CHECK:   IERT.RunTimeResources
// CHECK:       usedExecutors
// CHECK:           IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:           IERT.ExecutorResource 1 of "NCE_Cluster"

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
