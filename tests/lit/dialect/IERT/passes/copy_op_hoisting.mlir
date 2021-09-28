// RUN: vpux-opt --split-input-file --copy-op-hoisting %s | FileCheck %s
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 200704 + d1 * 1792 + d2 * 16 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 401408 + d1 * 3584 + d2 * 32 + d3)>

func @CopyOpHoisting(%arg0: memref<1x16x112x112xf16, #NHWC, #map0>, %arg1: memref<1x16x112x112xf16, #NHWC, #map0>, %arg2: memref<1x32x112x112xf16, #NHWC, #map1>) -> memref<1x32x112x112xf16, #NHWC, #map1> {
  %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC, #map1>

  %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN"> 
  %2 = IERT.Copy inputs(%arg1 : memref<1x16x112x112xf16, #NHWC, #map0>) outputs(%1 : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) -> memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">

  %3 = IERT.SubView %0 [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  %4 = IERT.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, #map0>) outputs(%3 : memref<1x16x112x112xf16, #NHWC, #map1>) -> memref<1x16x112x112xf16, #NHWC, #map1>

  %5 = IERT.SubView %0 [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  %6 = IERT.Copy inputs(%2 : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) outputs(%5 : memref<1x16x112x112xf16, #NHWC, #map1>) -> memref<1x16x112x112xf16, #NHWC, #map1>

  %7 = IERT.ConcatView inputs(%4, %6 : memref<1x16x112x112xf16, #NHWC, #map1>, memref<1x16x112x112xf16, #NHWC, #map1>) outputs(%0 : memref<1x32x112x112xf16, #NHWC, #map1>) -> memref<1x32x112x112xf16, #NHWC, #map1>
  %8 = IERT.Copy inputs(%7 : memref<1x32x112x112xf16, #NHWC, #map1>) outputs(%arg2 : memref<1x32x112x112xf16, #NHWC, #map1>) -> memref<1x32x112x112xf16, #NHWC, #map1>

  return %8 : memref<1x32x112x112xf16, #NHWC, #map1>

  // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x32x112x112xf16, #NHWC, #map1>
  // CHECK: [[VAR1:%.*]] = IERT.SubView 
  // CHECK-SAME:    [[VAR0]] [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC, #map1> to memref<1x16x112x112xf16, #NHWC, #map1>
  // CHECK: [[VAR2:%.*]] = IERT.Copy
  // CHECK-SAME:    inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, #map0, "CMX_NN">) outputs([[VAR1]] : memref<1x16x112x112xf16, #NHWC, #map1>)
}
