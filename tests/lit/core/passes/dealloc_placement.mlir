// RUN: vpux-opt --split-input-file --dealloc-placement %s | FileCheck %s

// -----

func @LinearGraph(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16, "DDR">
    %1 = IERT.SoftMax {axisInd = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">

    %2 = memref.alloc() : memref<1x1000xf16, "DDR">
    %3 = IERT.SoftMax {axisInd = 1 : i32} inputs(%1 : memref<1x1000xf16, "DDR">) outputs(%2 : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">

    %4 = memref.alloc() : memref<1x1000xf16, "DDR">
    %5 = IERT.SoftMax {axisInd = 1 : i32} inputs(%3 : memref<1x1000xf16, "DDR">) outputs(%4 : memref<1x1000xf16, "DDR">) -> memref<1x1000xf16, "DDR">

    %6 = IERT.Copy inputs(%2 : memref<1x1000xf16, "DDR">) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>

    return %6 : memref<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: [[VAR1:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: memref.dealloc [[VAR0]]
    // CHECK: [[VAR2:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: memref.dealloc [[VAR2]]
    // CHECK: [[VAR3:%.*]] = IERT.Copy
    // CHECK: memref.dealloc [[VAR1]]
    // CHECK: return [[VAR3]] : memref<1x1000xf16>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
    %0 = linalg.reshape %arg0 [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>

    %1 = memref.alloc() : memref<1x512xf32>
    %2 = IERT.SoftMax {axisInd = 1 : i32} inputs(%0 : memref<1x512xf32>) outputs(%1 : memref<1x512xf32>) -> memref<1x512xf32>

    %3 = linalg.reshape %2 [#map0, #map1] : memref<1x512xf32> into memref<1x512x1x1xf32>
    %4 = IERT.Copy inputs(%3 : memref<1x512x1x1xf32>) outputs(%arg1 : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>

    return %4 : memref<1x512x1x1xf32>

    // CHECK: linalg.reshape
    // CHECK: [[VAR1:%.*]] = memref.alloc()
    // CHECK: IERT.SoftMax
    // CHECK: linalg.reshape
    // CHECK: [[VAR2:%.*]] = IERT.Copy
    // CHECK: memref.dealloc [[VAR1]]
    // CHECK: return [[VAR2]] : memref<1x512x1x1xf32>
}
