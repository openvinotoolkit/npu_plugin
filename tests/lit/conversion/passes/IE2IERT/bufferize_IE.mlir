// RUN: vpux-opt --split-input-file --bufferize-IE %s | FileCheck %s

func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.buffer_cast %arg0 : memref<1x1000xf16>
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x1000xf16>
    // CHECK: [[VAR2:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x1000xf16>)
    // CHECK: [[VAR3:%.*]] = memref.tensor_load [[VAR2]] : memref<1x1000xf16>
    // CHECK: return [[VAR3]] : tensor<1x1000xf16>
}

// -----

func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = IE.Constant tensor<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK: [[VAR0:%.*]] = IERT.Constant memref<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf32>
    // CHECK: [[VAR1:%.*]] = memref.tensor_load [[VAR0]]
    // CHECK: return [[VAR1]] : tensor<1x2x2x2xf16>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = linalg.tensor_reshape %arg0 [#map0, #map1] : tensor<1x512x1x1xf32> into tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK: [[VAR0:%.*]] = memref.buffer_cast %arg0 : memref<1x512x1x1xf32>
    // CHECK: [[VAR1:%.*]] = linalg.reshape [[VAR0]] [#map0, #map1] : memref<1x512x1x1xf32> into memref<1x512xf32>
    // CHECK: [[VAR2:%.*]] = memref.tensor_load [[VAR1]]
    // CHECK: return [[VAR2]] : tensor<1x512xf32>
}

// -----

// CHECK:       #map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 8 + d2 * 2 + d3)>
// CHECK:       #map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 8 + d2 * 2 + d3 + 48)>

func @Split(%tensor: tensor<2x6x4x2xf32>) -> (tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>) {
    %0:2 = IE.Split(%tensor) {num_splits = 2 : i32, axis_value = 0 : si32} : tensor<2x6x4x2xf32> -> tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>
    return %0#0, %0#1 : tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>

    // CHECK-NOT:   IE.Split
    // CHECK:       [[BUFFER:%.*]] = memref.buffer_cast %arg0 : memref<2x6x4x2xf32>
    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x6x4x2xf32>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x6x4x2xf32>

    // CHECK:       [[VAR2:%.*]] = memref.subview [[BUFFER]][0, 0, 0, 0] [1, 6, 4, 2] [1, 1, 1, 1] : memref<2x6x4x2xf32> to memref<1x6x4x2xf32, #map0>
    // CHECK:       [[VAR4:%.*]] = IERT.Copy inputs([[VAR2]] : memref<1x6x4x2xf32, #map0>) outputs([[VAR0]] : memref<1x6x4x2xf32>) -> memref<1x6x4x2xf32>
    // CHECK:       [[VAR3:%.*]] = memref.subview [[BUFFER]][1, 0, 0, 0] [1, 6, 4, 2] [1, 1, 1, 1] : memref<2x6x4x2xf32> to memref<1x6x4x2xf32, #map1>
    // CHECK:       [[VAR5:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x6x4x2xf32, #map1>) outputs([[VAR1]] : memref<1x6x4x2xf32>) -> memref<1x6x4x2xf32>

    // CHECK:       [[OUT0:%.*]] = memref.tensor_load [[VAR4]] : memref<1x6x4x2xf32>
    // CHECK:       [[OUT1:%.*]] = memref.tensor_load [[VAR5]] : memref<1x6x4x2xf32>
    // CHECK:       return [[OUT0]], [[OUT1]] : tensor<1x6x4x2xf32>, tensor<1x6x4x2xf32>
}

// -----

func @Concat(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
  %0 = IE.Concat(%arg0, %arg1) {axis = 1 : si32} : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
  return %0 : tensor<1x4x3x4xf32>

  // CHECK-NOT:   IE.Concat
  // CHECK: [[VAR0:%.*]] = memref.buffer_cast %arg0 : memref<1x2x3x4xf32>
  // CHECK: [[VAR1:%.*]] = memref.buffer_cast %arg1 : memref<1x2x3x4xf32>
  // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x4x3x4xf32>
  // CHECK: [[VAR3:%.*]] = memref.subview [[VAR2]][0, 0, 0, 0] [1, 2, 3, 4] [1, 1, 1, 1] : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, #map0>
  // CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x2x3x4xf32>) outputs([[VAR3]] : memref<1x2x3x4xf32, #map0>) -> memref<1x2x3x4xf32, #map0>
  // CHECK: [[VAR5:%.*]] = memref.subview [[VAR2]][0, 2, 0, 0] [1, 2, 3, 4] [1, 1, 1, 1] : memref<1x4x3x4xf32> to memref<1x2x3x4xf32, #map1>
  // CHECK: [[VAR6:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x2x3x4xf32>) outputs([[VAR5]] : memref<1x2x3x4xf32, #map1>) -> memref<1x2x3x4xf32, #map1>
  // CHECK: [[VAR7:%.*]] = IERT.ConcatView inputs([[VAR4]], [[VAR6]] : memref<1x2x3x4xf32, #map0>, memref<1x2x3x4xf32, #map1>) outputs([[VAR2]] : memref<1x4x3x4xf32>) -> memref<1x4x3x4xf32>
  // CHECK: [[VAR8:%.*]] = memref.tensor_load [[VAR7]] : memref<1x4x3x4xf32>
  // CHECK: return [[VAR8]] : tensor<1x4x3x4xf32>
}
