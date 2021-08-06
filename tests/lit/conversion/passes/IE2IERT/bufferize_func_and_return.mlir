// RUN: vpux-opt --split-input-file --bufferize-func-and-return %s | FileCheck %s

// CHECK: func @SingleLayer(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = unrealized_conversion_cast %arg0 : tensor<1x1000xf16> to memref<1x1000xf16>
    %1 = memref.alloc() : memref<1x1000xf16>
    %2 = IERT.SoftMax {axisInd = 1} inputs(%0 : memref<1x1000xf16>) outputs(%1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %3 = unrealized_conversion_cast %2 : memref<1x1000xf16> to tensor<1x1000xf16>
    return %3 : tensor<1x1000xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x1000xf16>
}

// -----

// CHECK: func @OnlyOneOutput() -> memref<1x2x2x2xf16> {
func @OnlyOneOutput() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare memref<1x2x2x2xf16> = #const.Content<dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]>
    %1 = unrealized_conversion_cast %0 : memref<1x2x2x2xf16> to tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x2x2x2xf16>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 12 + d2 * 4 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 12 + d2 * 4 + d3 + 24)>

// CHECK: func @TwoInputs(%arg0: memref<1x2x3x4xf32>, %arg1: memref<1x2x3x4xf32>) -> memref<1x4x3x4xf32> {
func @TwoInputs(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf32> to memref<1x2x3x4xf32>
    %1 = unrealized_conversion_cast %arg1 : tensor<1x2x3x4xf32> to memref<1x2x3x4xf32>
    %2 = memref.alloc() : memref<1x4x3x4xf32>
    %3 = memref.subview %2[0, 0, 0, 0] [1, 2, 3, 4] [1, 1, 1, 1] : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, #map0>
    %4 = IERT.Copy inputs(%0 : memref<1x2x3x4xf32>) outputs(%3 : memref<1x2x3x4xf32, #map0>) -> memref<1x2x3x4xf32, #map0>
    %5 = memref.subview %2[0, 2, 0, 0] [1, 2, 3, 4] [1, 1, 1, 1] : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, #map1>
    %6 = IERT.Copy inputs(%1 : memref<1x2x3x4xf32>) outputs(%5 : memref<1x2x3x4xf32, #map1>) -> memref<1x2x3x4xf32, #map1>
    %7 = IERT.ConcatView inputs(%4, %6 : memref<1x2x3x4xf32, #map0>, memref<1x2x3x4xf32, #map1>) outputs(%2 : memref<1x4x3x4xf32>) -> memref<1x4x3x4xf32>
    %8 = unrealized_conversion_cast %7 : memref<1x4x3x4xf32> to tensor<1x4x3x4xf32>
    return %8 : tensor<1x4x3x4xf32>

    // CHECK: return
    // CHECK-SAME: memref<1x4x3x4xf32>
}
