// RUN: vpux-opt --split-input-file --bufferize-IE %s | FileCheck %s

func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.buffer_cast %arg0 : memref<1x1000xf16>
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x1000xf16>
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]]) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>
    // CHECK: [[VAR2:%.*]] = memref.tensor_load [[VAR1]] : memref<1x1000xf16>
    // CHECK: return [[VAR2]] : tensor<1x1000xf16>
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
