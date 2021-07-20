// RUN: vpux-opt -vpux-edsl-shave-pipeline -canonicalize %s | FileCheck %s

func @edsl_conv(%arg0: memref<3x3x3x8xf16>, %arg1: memref<1x64x64x3xf16>, %arg2: memref<1x64x64x8xf16>) -> memref<1x64x64x8xf16> {
  %cst = constant 0.000000e+00 : f16
  %0 = memref.alloc() : memref<1x66x66x3xf16>
  %1 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (66, 66, 3) reduce ("assign") -> (memref<1x66x66x3xf16>) {
    %4 = pxa.reduce assign %cst, %0[0, %arg3, %arg4, %arg5] : memref<1x66x66x3xf16>
    affine.yield %4 : memref<1x66x66x3xf16>
  }
  %2 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (64, 64, 3) reduce ("assign") -> (memref<1x66x66x3xf16>) {
    %4 = pxa.load %arg1[0, %arg3, %arg4, %arg5] : memref<1x64x64x3xf16>
    %5 = pxa.reduce assign %4, %1[0, %arg3 + 1, %arg4 + 1, %arg5] : memref<1x66x66x3xf16>
    affine.yield %5 : memref<1x66x66x3xf16>
  }
  %3 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (64, 64, 8) reduce ("assign") -> (memref<1x64x64x8xf16>) {
    %4 = pxa.reduce assign %cst, %arg2[0, %arg3, %arg4, %arg5] : memref<1x64x64x8xf16>
    %5 = affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (3, 3, 3) reduce ("assign") -> (memref<1x64x64x8xf16>) {
      %6 = pxa.load %2[0, %arg7 + %arg3, %arg8 + %arg4, %arg6] : memref<1x66x66x3xf16>
      %7 = pxa.load %arg0[%arg7, %arg8, %arg6, %arg5] : memref<3x3x3x8xf16>
      %8 = mulf %6, %7 : f16
      %9 = pxa.reduce addf %8, %4[0, %arg3, %arg4, %arg5] : memref<1x64x64x8xf16>
      affine.yield %9 : memref<1x64x64x8xf16>
    }
    affine.yield %5 : memref<1x64x64x8xf16>
  }
  return %3 : memref<1x64x64x8xf16>
}


// CHECK:       #map0 = affine_map<(d0) -> (d0 * 3072)>
// CHECK:       #map1 = affine_map<(d0) -> (d0 * 3168 + 201)>
// CHECK:       #map2 = affine_map<(d0) -> (d0 * 3168)>
// CHECK:       #map3 = affine_map<(d0) -> (0)>
// CHECK:       #map4 = affine_map<(d0) -> (d0 * 8192)>
// CHECK-LABEL: func @edsl_conv
// CHECK:         memref.alloc() : memref<1x66x66x3xf16>
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      inits = [unit, 0.000000e+00 : f16],
// CHECK-SAME:      kernel = @edsl_conv_kernel_1, middles = [], outers = [4],
// CHECK-SAME:      transfers = [{baseMap = #map0, dir = "IN", stage = "MIDDLE"}, {baseMap = #map1, dir = "OUT", stage = "MIDDLE"}]
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      inits = [unit, unit, unit],
// CHECK-SAME:      kernel = @edsl_conv_kernel_2, middles = [], outers = [4],
// CHECK-SAME:      transfers = [{baseMap = #map2, dir = "IN", stage = "MIDDLE"}, {baseMap = #map3, dir = "IN", stage = "MIDDLE"}, {baseMap = #map4, dir = "OUT", stage = "MIDDLE"}]
// CHECK:       module @kernels
// CHECK:         func @edsl_conv_kernel_1(%{{.*}}: index, %{{.*}}: memref<1x16x64x3xf16>, %{{.*}}: memref<1x16x64x3xf16>) -> memref<1x16x64x3xf16>
// CHECK:         func @edsl_conv_kernel_2(%{{.*}}: index, %{{.*}}: memref<1x18x66x3xf16>, %{{.*}}: memref<3x3x3x8xf16>, %{{.*}}: memref<1x16x64x8xf16>) -> memref<1x16x64x8xf16>
