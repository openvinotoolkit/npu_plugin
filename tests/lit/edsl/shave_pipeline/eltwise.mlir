// RUN: vpux-opt -vpux-edsl-shave-pipeline -canonicalize %s | FileCheck %s

func @eltwise_sum(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x256x16xf16>, %out: memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16> {
  %1 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (256, 256, 16) reduce ("assign") -> (memref<1x256x256x16xf16>) {
    %2 = pxa.load %arg1[0, %arg2, %arg3, %arg4] : memref<1x256x256x16xf16>
    %3 = pxa.load %arg0[0, %arg2, %arg3, %arg4] : memref<1x256x256x16xf16>
    %4 = addf %2, %3 : f16
    %5 = pxa.reduce assign %4, %out[0, %arg2, %arg3, %arg4] : memref<1x256x256x16xf16>
    affine.yield %5 : memref<1x256x256x16xf16>
  }
  return %1 : memref<1x256x256x16xf16>
}

// #map = affine_map<(d0) -> (d0 * 8192)>
// CHECK-LABEL: func @eltwise_sum
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      kernel = @eltwise_sum_kernel_0, middles = [], outers = [128],
// CHECK-SAME:      transfers = [{baseMap = #map, dir = "IN", stage = "MIDDLE"}, {baseMap = #map, dir = "IN", stage = "MIDDLE"}, {baseMap = #map, dir = "OUT", stage = "MIDDLE"}]
// CHECK:       module @kernels
// CHECK:         func @eltwise_sum_kernel_0(
// CHECK-SAME:      %{{.*}}: index, %{{.*}}: memref<1x2x256x16xf16>, %{{.*}}: memref<1x2x256x16xf16>, %{{.*}}: memref<1x2x256x16xf16>) -> memref<1x2x256x16xf16> {
