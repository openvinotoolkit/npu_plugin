// RUN: vpux-opt -vpux-edsl-shave-pipeline -canonicalize %s | FileCheck %s

func @gather(%arg0: memref<10x20x30x40xf32>, %arg1: memref<4xi64>, %arg2: memref<4x20x30x40xf32>) -> memref<4x20x30x40xf32> {
  %0 = affine.parallel (%arg4, %arg5, %arg6, %arg7) = (0, 0, 0, 0) to (4, 20, 30, 40) reduce ("assign") -> (memref<4x20x30x40xf32>) {
    %1 = pxa.load %arg1[%arg4] : memref<4xi64>
    %2 = index_cast %1 : i64 to index
    %3 = memref.load %arg0[%2, %arg5, %arg6, %arg7] : memref<10x20x30x40xf32>
    %4 = pxa.reduce assign %3, %arg2[%arg4, %arg5, %arg6, %arg7] : memref<4x20x30x40xf32>
    affine.yield %4 : memref<4x20x30x40xf32>
  }
  return %0 : memref<4x20x30x40xf32>
}

// CHECK:       #map0 = affine_map<() -> (0)>
// CHECK:       #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK:       #map2 = affine_map<(d0, d1) -> (d0 * 24000 + d1 * 12000)>
// CHECK-LABEL: func @gather
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      kernel = @gather_kernel_0, middles = [], outers = [4, 2],
// CHECK-SAME:      transfers = [{baseMap = #map0, dir = "IN", stage = "ALL"}, {baseMap = #map1, dir = "IN", stage = "MIDDLE"}, {baseMap = #map2, dir = "OUT", stage = "MIDDLE"}]
// CHECK:       module @kernels
// CHECK:         func @gather_kernel_0(
// CHECK-SAME:      %{{.*}}: index, %{{.*}}: index, %{{.*}}: memref<10x20x30x40xf32>, %{{.*}}: memref<1xi64>, %{{.*}}: memref<1x10x30x40xf32>) -> memref<1x10x30x40xf32> {


