// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --lower-IE-to-IERT %s | FileCheck %s

//
// The 'lower-IE-to-IERT' pass:
//
//   * Fully replaces IE Dialect with IERT Dielect.
//   * Changes all Values types from `tensor` to `memref`.
//   * Changes Function results tensors to arguments.
//   * Inserts `IERT.Copy` for `const.Declare` as result case.
//

// CHECK: func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1000xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1000xf16>)

    // CHECK: [[VAR2:%.*]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)

    // CHECK: return [[VAR2]] : memref<1x1000xf16>
}

// -----

// CHECK: func @ConstantLayer([[ARG0:%.*]]: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> = #const.Content<dense<1.0> : tensor<1x2x2x2xf16>>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK: [[VAR0:%.*]] = const.Declare memref<1x2x2x2xf16> = #const.Content<dense<1.000000e+00> : tensor<1x2x2x2xf16>>
    // CHECK: [[VAR1:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x2x2x2xf16>) outputs([[ARG0]] : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    // CHECK: return [[VAR1]] : memref<1x2x2x2xf16>
}

// -----

// CHECK: func @Reshape([[ARG0:%.*]]: memref<1x512x1x1xf32>, [[ARG1:%.*]]: memref<1x512xf32>) -> memref<1x512xf32> {
func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK: [[VAR0:%.*]] = IERT.GenericReshape inputs([[ARG0]] : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    // CHECK: [[VAR1:%.*]] = IERT.Copy inputs([[VAR0]] : memref<1x512xf32>) outputs([[ARG1]] : memref<1x512xf32>) -> memref<1x512xf32>
    // CHECK: return [[VAR1]] : memref<1x512xf32>
}

// -----

// CHECK: func @ReshapeInGraph([[ARG0:%.*]]: memref<1x512x1x1xf32>, [[ARG1:%.*]]: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
func @ReshapeInGraph(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x512xf32> -> tensor<1x512xf32>
    %2 = IE.Reshape(%1) { shape_value = [1, 512, 1, 1] } : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    return %2 : tensor<1x512x1x1xf32>

    // CHECK: [[VAR0:%.*]] = IERT.GenericReshape inputs([[ARG0]] : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x512xf32>
    // CHECK: [[VAR2:%.*]] = IERT.SoftMax
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[VAR0]] : memref<1x512xf32>)
    // CHECK-SAME:              outputs([[VAR1]] : memref<1x512xf32>)

    // CHECK: [[VAR3:%.*]] = IERT.GenericReshape inputs([[VAR2]] : memref<1x512xf32>) -> memref<1x512x1x1xf32>
    // CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x512x1x1xf32>) outputs([[ARG1]] : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>
    // CHECK: return [[VAR4]] : memref<1x512x1x1xf32>
}
