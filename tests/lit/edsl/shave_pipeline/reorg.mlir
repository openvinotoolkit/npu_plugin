// RUN: vpux-opt -vpux-edsl-shave-pipeline -canonicalize %s | FileCheck %s

func @reorg_yolo(%arg0: memref<1x64x26x26xf16>, %arg1: memref<1x256x13x13xf16>) -> memref<1x256x13x13xf16> {
  %cst = constant 0.000000e+00 : f32
  %0 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (256, 13, 13) reduce ("assign") -> (memref<1x256x13x13xf16>) {
    %2 = fptrunc %cst : f32 to f16
    %3 = pxa.reduce assign %2, %arg1[0, %arg2, %arg3, %arg4] : memref<1x256x13x13xf16>
    affine.yield %3 : memref<1x256x13x13xf16>
  }
  %1 = affine.parallel (%arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0) to (64, 2, 2, 13, 13) reduce ("assign") -> (memref<1x256x13x13xf16>) {
    %2 = pxa.load %arg0[0, %arg2, %arg4 + %arg5 * 2, %arg3 + %arg6 * 2] : memref<1x64x26x26xf16>
    %3 = pxa.reduce assign %2, %0[0, %arg2 + %arg3 * 64 + %arg4 * 128, %arg5, %arg6] : memref<1x256x13x13xf16>
    affine.yield %3 : memref<1x256x13x13xf16>
  }
  return %1 : memref<1x256x13x13xf16>
}

// CHECK:       #map0 = affine_map<(d0, d1) -> (d0 * 10816 + d1 * 26)>
// CHECK:       #map1 = affine_map<(d0, d1) -> (d0 * 2704 + d1 * 21632)>
// CHECK-LABEL: func @reorg_yolo
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      inits = [unit, 0.000000e+00 : f32],
// CHECK-SAME:      kernel = @reorg_yolo_kernel_1, middles = [], outers = [4, 2],
// CHECK-SAME:      transfers = [{baseMap = #map0, dir = "IN", stage = "MIDDLE"}, {baseMap = #map1, dir = "OUT", stage = "MIDDLE"}]
// CHECK:       module @kernels
// CHECK:         func @reorg_yolo_kernel_1(%{{.*}}: index, %{{.*}}: index, %{{.*}}: memref<1x16x25x26xf16>, %{{.*}}: memref<1x80x13x13xf16>) -> memref<1x80x13x13xf16>
