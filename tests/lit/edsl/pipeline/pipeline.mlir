// RUN: vpux-opt --lower-TILE-to-VPUIP %s | FileCheck %s 

module {

func @main(%arg0: tensor<1x256x256x16xf16>, %arg1: tensor<1x256x256x16xf16>) -> tensor<1x256x256x16xf16> {
  %0 = tile.add %arg0, %arg1 : (tensor<1x256x256x16xf16>, tensor<1x256x256x16xf16>) -> tensor<1x256x256x16xf16>
  return %0 : tensor<1x256x256x16xf16>
}

}

// CHECK: #map = affine_map<(d0) -> (d0 * 8192)>
// CHECK: module
// CHECK:   func @main(%{{.*}}: memref<1x256x256x16xf16>, %{{.*}}: memref<1x256x256x16xf16>, %{{.*}}: memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16> {
// CHECK:     VPUIP.EdslUPA {
// CHECK-SAME:  inits = [unit, unit, unit],
// CHECK-SAME:  kernel = @main_kernel_0,
// CHECK-SAME:  middles = [],
// CHECK-SAME:  outers = [128],
// CHECK-SAME:  transfers = [{baseMap = #map, dir = "IN", stage = "MIDDLE"}, {baseMap = #map, dir = "IN", stage = "MIDDLE"}, {baseMap = #map, dir = "OUT", stage = "MIDDLE"}]}
// CHECK-SAME:  inputs(%arg0, %arg1 : memref<1x256x256x16xf16>, memref<1x256x256x16xf16>)
// CHECK-SAME:  outputs(%arg2 : memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16>
// CHECK:     return
// CHECK:   module @kernels
// CHECK:     func @main_kernel_0(%{{.*}}: index, %{{.*}}: memref<1x2x256x16xf16>, %{{.*}}: memref<1x2x256x16xf16>, %{{.*}}: memref<1x2x256x16xf16>)
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:       return
