// RUN: vpux-opt -split-input-file -convert-IE-to-VPUIP %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:       VPUIP.Graph
// CHECK-SAME:      entryPoint = @main, identifier = "SingleLayer"
// CHECK-SAME:      options = []
// CHECK-SAME:      nn_cmx_slice_amount = 1
// CHECK-SAME:      upa_shaves = 1
IE.CNNNetwork {
    entryPoint = @main, netName = "SingleLayer"
} inputsInfo {
    // CHECK:       VPUIP.TensorInfo
    // CHECK-SAME:      layout = #[[MAP]]
    // CHECK-SAME:      name = "data"
    // CHECK-SAME:      precision = f32
    IE.DataInfo {layout = #IE<"Layout:NC">, name = "data", precision = f32}
} outputsInfo {
    // CHECK:       VPUIP.TensorInfo
    // CHECK-SAME:      layout = #[[MAP]]
    // CHECK-SAME:      name = "prob"
    // CHECK-SAME:      precision = f32
    IE.DataInfo {layout = #IE<"Layout:NC">, name = "prob", precision = f32}
}

// CHECK: func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>)
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK:       %0 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    // CHECK-NEXT:  VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)

    return %prob : tensor<1x1000xf16>
    // CHECK:       VPUIP.UPADMA inputs(%0 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)
    // CHECK-NEXT:  return
}
