// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --set-internal-memory-space="memory-space=DDR" %s | FileCheck %s

//
// The 'set-internal-memory-space' pass:
//
//   * Updates only Function bodies.
//   * Updates `alloc` Operation result Type.
//

func @MultipleAllocs(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    %1 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    IERT.SoftMax(%1, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    return

    // CHECK: [[VAR0:%.*]] = alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax(%arg0, [[VAR0]]) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16, "DDR">
    // CHECK: [[VAR1:%.*]] = alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]]) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[VAR1]], %arg1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16>
    // CHECK: return
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) {
    %0 = linalg.reshape %arg0 [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>
    %1 = alloc() : memref<1x512xf32>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x512xf32>, memref<1x512xf32>
    %2 = linalg.reshape %1 [#map0, #map1] : memref<1x512xf32> into memref<1x512x1x1xf32>
    linalg.copy(%2, %arg1) : memref<1x512x1x1xf32>, memref<1x512x1x1xf32>
    dealloc %1 : memref<1x512xf32>
    return

    // CHECK: [[VAR0:%.*]] = linalg.reshape %arg0 [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>
    // CHECK: [[VAR1:%.*]] = alloc() : memref<1x512xf32, "DDR">
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]]) {axisInd = 1 : i32} : memref<1x512xf32>, memref<1x512xf32, "DDR">
    // CHECK: [[VAR2:%.*]] = linalg.reshape [[VAR1]] [#map0, #map1] : memref<1x512xf32, "DDR"> into memref<1x512x1x1xf32, "DDR">
    // CHECK: linalg.copy([[VAR2]], %arg1) : memref<1x512x1x1xf32, "DDR">, memref<1x512x1x1xf32>
    // CHECK: dealloc [[VAR1]] : memref<1x512xf32, "DDR">
    // CHECK: return
}
