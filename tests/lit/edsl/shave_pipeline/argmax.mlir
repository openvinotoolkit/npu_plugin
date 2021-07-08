// RUN: vpux-opt -vpux-edsl-shave-pipeline -canonicalize %s | FileCheck %s

func @edsl_argmax(%arg0: memref<1x256x256x16xf16>, %out: memref<1x256x16xi32>) -> memref<1x256x16xi32> {
  %cst = constant 0xFC00 : f16
  %c0_i32 = constant 0 : i32
  %0 = memref.alloc() : memref<1x256x16xf16>
  %1 = affine.parallel (%arg1, %arg2) = (0, 0) to (256, 16) reduce ("assign") -> (memref<1x256x16xf16>) {
    %5 = pxa.reduce assign %cst, %0[0, %arg1, %arg2] : memref<1x256x16xf16>
    %6 = affine.parallel (%arg3) = (0) to (256) reduce ("assign") -> (memref<1x256x16xf16>) {
      %7 = pxa.load %arg0[0, %arg3, %arg1, %arg2] : memref<1x256x256x16xf16>
      %8 = pxa.reduce maxf %7, %5[0, %arg1, %arg2] : memref<1x256x16xf16>
      affine.yield %8 : memref<1x256x16xf16>
    }
    affine.yield %6 : memref<1x256x16xf16>
  }
  %3 = affine.parallel (%arg1, %arg2) = (0, 0) to (256, 16) reduce ("assign") -> (memref<1x256x16xi32>) {
    %5 = pxa.reduce assign %c0_i32, %out[0, %arg1, %arg2] : memref<1x256x16xi32>
    affine.yield %5 : memref<1x256x16xi32>
  }
  %4 = affine.parallel (%arg1) = (0) to (256) reduce ("assign") -> (memref<1x256x16xi32>) {
    %5 = index_cast %arg1 : index to i32
    %6 = affine.parallel (%arg2, %arg3) = (0, 0) to (256, 16) reduce ("assign") -> (memref<1x256x16xi32>) {
      %7 = pxa.load %arg0[0, %arg1, %arg2, %arg3] : memref<1x256x256x16xf16>
      %8 = pxa.load %1[0, %arg2, %arg3] : memref<1x256x16xf16>
      %9 = cmpf "oeq", %7, %8 : f16
      %10 = select %9, %5, %c0_i32 : i32
      %11 = pxa.reduce maxu %10, %3[0, %arg2, %arg3] : memref<1x256x16xi32>
      affine.yield %11 : memref<1x256x16xi32>
    }
    affine.yield %6 : memref<1x256x16xi32>
  }
  return %4 : memref<1x256x16xi32>
}

// CHECK:       #map0 = affine_map<(d0) -> (d0 * 64)>
// CHECK:       #map1 = affine_map<(d0) -> (d0 * 16384)>
// CHECK:       #map2 = affine_map<() -> (0)>
// CHECK-LABEL: func @edsl_argmax
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      kernel = @edsl_argmax_kernel_0
// CHECK-SAME:      middles = [], outers = [64],
// CHECK-SAME:      transfers = [{baseMap = #map0, dir = "IN", stage = "MIDDLE"}, {baseMap = #map0, dir = "OUT", stage = "MIDDLE"}]
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      kernel = @edsl_argmax_kernel_2
// CHECK-SAME:      middles = [64], outers = [],
// CHECK-SAME:      transfers =  [{baseMap = #map1, dir = "IN", stage = "MIDDLE"}, {baseMap = #map2, dir = "IN", stage = "OUTER"}, {baseMap = #map2, dir = "OUT", stage = "OUTER"}]
// CHECK:       module @kernels
// CHECK:         func @edsl_argmax_kernel_0(%{{.*}}: index, %{{.*}}: memref<1x256x4x16xf16>, %{{.*}}: memref<1x4x16xf16>) -> memref<1x4x16xf16>
// CHECK:         func @edsl_argmax_kernel_2(%{{.*}}: index, %{{.*}}: memref<1x4x256x16xf16>, %{{.*}}: memref<1x256x16xf16>, %{{.*}}: memref<1x256x16xi32>) -> memref<1x256x16xi32>
