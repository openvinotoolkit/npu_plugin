// RUN: vpux-opt --split-input-file --dealloc-placement %s | FileCheck %s

// -----

func @LinearGraph(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16, "DDR">

    %1 = memref.alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">

    %2 = memref.alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%1, %2) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">

    IERT.Copy(%2, %arg1) : memref<1x1000xf16, "DDR">, memref<1x1000xf16>

    return %arg1 : memref<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: [[VAR1:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: memref.dealloc [[VAR0]]
    // CHECK: [[VAR2:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: memref.dealloc [[VAR1]]
    // CHECK: IERT.Copy
    // CHECK: memref.dealloc [[VAR2]]
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
    %0 = linalg.reshape %arg0 [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>

    %1 = memref.alloc() : memref<1x512xf32>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x512xf32>, memref<1x512xf32>

    %2 = linalg.reshape %1 [#map0, #map1] : memref<1x512xf32> into memref<1x512x1x1xf32>
    IERT.Copy(%2, %arg1) : memref<1x512x1x1xf32>, memref<1x512x1x1xf32>

    return %arg1 : memref<1x512x1x1xf32>

    // CHECK: linalg.reshape
    // CHECK: [[VAR1:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: linalg.reshape
    // CHECK: IERT.Copy
    // CHECK: memref.dealloc [[VAR1]]
}
