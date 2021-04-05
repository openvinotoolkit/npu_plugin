// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --set-internal-memory-space="memory-space=DDR" %s | FileCheck %s

//
// The 'set-internal-memory-space' pass:
//
//   * Updates only Function bodies.
//   * Updates `memref.alloc` Operation result Type.
//

func @MultipleAllocs(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    %1 = memref.alloc() : memref<1x1000xf16>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    IERT.SoftMax(%1, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    return %arg1 : memref<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax(%arg0, [[VAR0]]) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16, "DDR">
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]]) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[VAR1]], %arg1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16>
    // CHECK: return %arg1 : memref<1x1000xf16>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
    %0 = linalg.reshape %arg0 [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>
    %1 = memref.alloc() : memref<1x512xf32>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x512xf32>, memref<1x512xf32>
    %2 = linalg.reshape %1 [#map0, #map1] : memref<1x512xf32> into memref<1x512x1x1xf32>
    linalg.copy(%2, %arg1) : memref<1x512x1x1xf32>, memref<1x512x1x1xf32>
    memref.dealloc %1 : memref<1x512xf32>
    return %arg1 : memref<1x512x1x1xf32>

    // CHECK: [[VAR0:%.*]] = linalg.reshape %arg0 [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x512xf32, "DDR">
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]]) {axisInd = 1 : i32} : memref<1x512xf32>, memref<1x512xf32, "DDR">
    // CHECK: [[VAR2:%.*]] = linalg.reshape [[VAR1]] [#map0, #map1] : memref<1x512xf32, "DDR"> into memref<1x512x1x1xf32, "DDR">
    // CHECK: linalg.copy([[VAR2]], %arg1) : memref<1x512x1x1xf32, "DDR">, memref<1x512x1x1xf32>
    // CHECK: memref.dealloc [[VAR1]] : memref<1x512xf32, "DDR">
    // CHECK: return %arg1 : memref<1x512x1x1xf32>
}
