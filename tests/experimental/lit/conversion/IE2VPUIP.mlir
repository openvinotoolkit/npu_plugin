// RUN: vpux-opt -split-input-file -convert-IE-to-VPUIP %s | FileCheck %s

// CHECK: #NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: VPUIP.Graph "SingleLayer" at @main
// CHECK-SAME:      options : "NONE"
// CHECK-SAME:      item = "SHAVE_UPA", number = 1
// CHECK-SAME:      item = "NCE_Cluster", number = 1
IE.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        // CHECK: VPUIP.TensorInfo "data", f32, #NC
        IE.DataInfo "data", f32, "NC"
    }
    outputsInfo : {
        // CHECK: VPUIP.TensorInfo "prob", f32, #NC
        IE.DataInfo "prob", f32, "NC"
    }

// CHECK: func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>)
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK:       %0 = VPUIP.DeclareTensor "VPU_DDR_Heap"
    // CHECK-NEXT:  VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)

    return %prob : tensor<1x1000xf16>
    // CHECK:       VPUIP.UPADMA inputs(%0 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)
    // CHECK-NEXT:  return
}

// -----

// CHECK-LABEL: VPUIP.Graph "ConstantLayer" at @main
IE.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "output", f32, "NCHW"
    }

// CHECK: func @main(%arg0: memref<1x2x2x2xf16>)
func @main() -> tensor<1x2x2x2xf16> {
    %0 = constant
        dense<[
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0]
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0]
                ]
            ]
        ]> : tensor<1x2x2x2xf16>
    // CHECK:       %[[VAR:.*]] = VPUIP.DeclareConstantTensorOp
    // CHECK-SAME:      memref<1x2x2x2xf16>

    return %0 : tensor<1x2x2x2xf16>
    // CHECK:       VPUIP.UPADMA inputs(%[[VAR]] : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>)
    // CHECK-NEXT:  return
}
