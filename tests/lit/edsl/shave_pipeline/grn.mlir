// RUN: vpux-opt -vpux-edsl-shave-pipeline -canonicalize %s | FileCheck %s

func @grn(%arg0: memref<1x224x128x24xf16>, %out: memref<1x224x128x24xf16>) -> memref<1x224x128x24xf16> {
  %cst = constant 0.000000e+00 : f16
  %cst_0 = constant 1.000000e+00 : f16
  %cst_1 = constant 1.001360e-05 : f16
  %cst_2 = constant 1.600000e+01 : f16
  %0 = affine.parallel (%arg1, %arg2) = (0, 0) to (224, 128) reduce ("assign") -> (memref<1x224x128x24xf16>) {
    %1 = memref.alloc() : memref<1x1x1x1xf32>
    %2 = fpext %cst : f16 to f32
    %3 = pxa.reduce assign %2, %1[0, 0, 0, 0] : memref<1x1x1x1xf32>
    %4 = affine.parallel (%arg3) = (0) to (24) reduce ("assign") -> (memref<1x1x1x1xf32>) {
      %15 = pxa.load %arg0[0, %arg1, %arg2, %arg3] : memref<1x224x128x24xf16>
      %16 = fpext %15 : f16 to f32
      %17 = mulf %16, %16 : f32
      %18 = pxa.reduce addf %17, %3[0, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.yield %18 : memref<1x1x1x1xf32>
    }
    %5 = pxa.load %4[0, 0, 0, 0] : memref<1x1x1x1xf32>
    %6 = fpext %cst_0 : f16 to f32
    %7 = addf %5, %6 : f32
    %8 = fpext %cst_1 : f16 to f32
    %9 = addf %7, %8 : f32
    %10 = fpext %cst_2 : f16 to f32
    %11 = divf %9, %10 : f32
    %12 = fptrunc %11 : f32 to f16
    %13 = math.sqrt %12 : f16
    %14 = affine.parallel (%arg5) = (0) to (24) reduce ("assign") -> (memref<1x224x128x24xf16>) {
      %15 = pxa.load %arg0[0, %arg1, %arg2, %arg5] : memref<1x224x128x24xf16>
      %16 = divf %15, %13 : f16
      %17 = pxa.reduce assign %16, %out[0, %arg1, %arg2, %arg5] : memref<1x224x128x24xf16>
      affine.yield %17 : memref<1x224x128x24xf16>
    }
    affine.yield %14 : memref<1x224x128x24xf16>
  }
  return %0 : memref<1x224x128x24xf16>
}


// CHECK:       #map = affine_map<(d0) -> (d0 * 12288)>
// CHECK-LABEL: func @grn
// CHECK:         VPUIP.EdslUPA
// CHECK-SAME:      kernel = @grn_kernel_0, middles = [], outers = [56],
// CHECK-SAME:      transfers = [{baseMap = #map, dir = "IN", stage = "MIDDLE"}, {baseMap = #map, dir = "OUT", stage = "MIDDLE"}]
// CHECK:       module @kernels
// CHECK:         func @grn_kernel_0(
// CHECK-SAME:      %{{.*}}: index, %{{.*}}: memref<1x4x128x24xf16>, %{{.*}}: memref<1x4x128x24xf16>) -> memref<1x4x128x24xf16>
