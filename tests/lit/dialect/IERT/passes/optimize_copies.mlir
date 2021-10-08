// RUN: vpux-opt --split-input-file --optimize-copies %s | FileCheck %s
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 200704 + d1 * 1792 + d2 * 16 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 401408 + d1 * 3584 + d2 * 32 + d3)>

func @OptimizeCopy(%arg0: memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">, %arg1: memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">, %arg2: memref<1x32x112x112xf16, #NHWC, #map1>) -> memref<1x32x112x112xf16, #NHWC, #map1> {

  %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC, #map1>

  %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, #map0>
  %2 = IERT.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) outputs(%1 : memref<1x16x112x112xf16, #NHWC, #map0>) -> memref<1x16x112x112xf16, #NHWC, #map0>
  %3 = IERT.SubView %0 [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  %4 = IERT.Copy inputs(%2 : memref<1x16x112x112xf16, #NHWC, #map0>) outputs(%3 : memref<1x16x112x112xf16, #NHWC, #map1>) -> memref<1x16x112x112xf16, #NHWC, #map1>

  %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, #map0>
  %6 = IERT.Copy inputs(%arg1 : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) outputs(%5 : memref<1x16x112x112xf16, #NHWC, #map0>) -> memref<1x16x112x112xf16, #NHWC, #map0>
  %7 = IERT.SubView %0 [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  %8 = IERT.Copy inputs(%6 : memref<1x16x112x112xf16, #NHWC, #map0>) outputs(%7 : memref<1x16x112x112xf16, #NHWC, #map1>) -> memref<1x16x112x112xf16, #NHWC, #map1>

  %9 = IERT.ConcatView inputs(%4, %8 : memref<1x16x112x112xf16, #NHWC, #map1>, memref<1x16x112x112xf16, #NHWC, #map1>) outputs(%0 : memref<1x32x112x112xf16, #NHWC, #map1>) -> memref<1x32x112x112xf16, #NHWC, #map1>
  %10 = IERT.Copy inputs(%9 : memref<1x32x112x112xf16, #NHWC, #map1>) outputs(%arg2 : memref<1x32x112x112xf16, #NHWC, #map1>) -> memref<1x32x112x112xf16, #NHWC, #map1>

  return %10 : memref<1x32x112x112xf16, #NHWC, #map1>

  // CHECK-NOT: memref.alloc() : memref<1x16x112x112xf16, #NHWC, #map0>
  // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xf16, #NHWC, #map1>
  // CHECK: [[VAR1:%.*]] = IERT.SubView [[VAR0]] [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  // CHECK: [[VAR2:%.*]] = IERT.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) outputs([[VAR1]] : memref<1x16x112x112xf16, #NHWC, #map1>) -> memref<1x16x112x112xf16, #NHWC, #map1>
  // CHECK: [[VAR3:%.*]] = IERT.SubView [[VAR0]] [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  // CHECK: [[VAR4:%.*]] = IERT.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) outputs([[VAR3]] : memref<1x16x112x112xf16, #NHWC, #map1>) -> memref<1x16x112x112xf16, #NHWC, #map1>

}
