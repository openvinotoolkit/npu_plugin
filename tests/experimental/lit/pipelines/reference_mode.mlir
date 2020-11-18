// RUN: vpux-opt -split-input-file -reference-mode %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:       VPUIP.Graph
// CHECK-SAME:      entryPoint = @main, identifier = "SingleLayer"
// CHECK-SAME:      options = [#VPUIP<"ExecutionFlag:DynamicBarriers">]
// CHECK-SAME:      nn_cmx_slice_amount = 1
// CHECK-SAME:      upa_shaves = 1
IE.CNNNetwork {
    entryPoint = @main, netName = "SingleLayer"
} inputsInfo {
    // CHECK:       VPUIP.TensorInfo
    // CHECK-SAME:      layout = #[[MAP]]
    // CHECK-SAME:      name = "input"
    // CHECK-SAME:      precision = f32
    IE.DataInfo {name = "input", precision = f32, layout = #IE<"Layout:NC">}
}
outputsInfo {
    // CHECK:       VPUIP.TensorInfo
    // CHECK-SAME:      layout = #[[MAP]]
    // CHECK-SAME:      name = "softmax"
    // CHECK-SAME:      precision = f32
    IE.DataInfo {name = "softmax", precision = f32, layout = #IE<"Layout:NC">}
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
