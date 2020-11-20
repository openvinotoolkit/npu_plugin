// RUN: vpux-opt -split-input-file -reference-mode %s | FileCheck %s

// CHECK: #NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:       VPUIP.Graph "SingleLayer" at @main
// CHECK-SAME:      options : "DynamicBarriers"
// CHECK-SAME:      nn_cmx_slice_amount = 1
// CHECK-SAME:      upa_shaves = 1
IE.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        // CHECK: VPUIP.TensorInfo "input", f32, #NC
        IE.DataInfo "input", f32, "NC"
    }
    outputsInfo : {
        // CHECK: VPUIP.TensorInfo "softmax", f32, #NC
        IE.DataInfo "softmax", f32, "NC"
    }

// CHECK: func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %0 : tensor<1x1000xf32>

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      isTrailingSWLayer
    // CHECK-SAME:      maxShaves = 1
    // CHECK-SAME:      inputs(%arg0 : memref<1x1000xf16>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1000xf16>)
    // CHECK-NEXT:  return
}
