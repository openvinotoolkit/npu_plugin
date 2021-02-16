// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Constant tensor<2xsi64> = dense<[4, 4]> : tensor<2xsi64>
    %1 = IE.Reshape(%arg0, %0) : tensor<4x4xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK-NOT:   IE.Constant
    // CHECK-NOT:   IE.Reshape
    // CHECK:       return %arg0
}

// -----

func @ConstFold() -> tensor<4x4xf32> {
    %0 = IE.Constant tensor<1x16xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.Constant tensor<2xsi64> = dense<[4, 4]> : tensor<2xsi64>
    %2 = IE.Reshape(%0, %1) : tensor<1x16xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>

    // CHECK:       %[[VAL0:.*]] = IE.Constant tensor<4x4xf32> = dense<1.000000e+00> : tensor<16xf32>
    // CHECK-NOT:   IE.Constant
    // CHECK-NOT:   IE.Reshape
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>

// CHECK: [[MAP2:#.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: [[MAP3:#.*]] = affine_map<(d0, d1, d2) -> (d2)>

func @UseLinalgReshapeCollapse(
        %arg0 : tensor<1x512x1x1xf32>,
        %arg1 : tensor<3x1x1xf32>,
        %arg2 : tensor<1x100x1x1xf32>) ->
            (tensor<1x512xf32>, tensor<3x1xf32>, tensor<1x100xf32>) {
    %0 = IE.Constant tensor<2xsi64> = dense<[1, -1]> : tensor<2xsi64>
    %1 = IE.Reshape(%arg0, %0) {special_zero} : tensor<1x512x1x1xf32>, tensor<2xsi64> -> tensor<1x512xf32>

    %2 = IE.Constant tensor<2xsi64> = dense<[-1, 0]> : tensor<2xsi64>
    %3 = IE.Reshape(%arg1, %2) {special_zero} : tensor<3x1x1xf32>, tensor<2xsi64> -> tensor<3x1xf32>

    %4 = IE.Constant tensor<2xsi64> = dense<[0, 100]> : tensor<2xsi64>
    %5 = IE.Reshape(%arg2, %4) {special_zero} : tensor<1x100x1x1xf32>, tensor<2xsi64> -> tensor<1x100xf32>

    return %1, %3, %5 : tensor<1x512xf32>, tensor<3x1xf32>, tensor<1x100xf32>

    // CHECK:       [[VAL0:%.*]] = linalg.tensor_reshape %arg0
    // CHECK-SAME:      [[MAP0]], [[MAP1]]
    // CHECK-SAME:      tensor<1x512x1x1xf32> into tensor<1x512xf32>

    // CHECK:       [[VAL1:%.*]] = linalg.tensor_reshape %arg1
    // CHECK-SAME:      [[MAP2]], [[MAP3]]
    // CHECK-SAME:      tensor<3x1x1xf32> into tensor<3x1xf32>

    // CHECK:       [[VAL2:%.*]] = linalg.tensor_reshape %arg2
    // CHECK-SAME:      [[MAP0]], [[MAP1]]
    // CHECK-SAME:      tensor<1x100x1x1xf32> into tensor<1x100xf32>

    // CHECK:       return [[VAL0]], [[VAL1]], [[VAL2]] : tensor<1x512xf32>, tensor<3x1xf32>, tensor<1x100xf32>
}

// -----

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1, d2, d3) -> (d1)>
// CHECK: [[MAP2:#.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>

func @UseLinalgReshapeExpand(
        %arg0 : tensor<2x2x3xf32>) ->
            tensor<2x2x1x3xf32> {
    %0 = IE.Constant tensor<4xsi64> = dense<[0, 0, 1, -1]> : tensor<4xsi64>
    %1 = IE.Reshape(%arg0, %0) {special_zero} : tensor<2x2x3xf32>, tensor<4xsi64> -> tensor<2x2x1x3xf32>
    return %1 : tensor<2x2x1x3xf32>

    // CHECK:       [[VAL0:%.*]] = linalg.tensor_reshape %arg0
    // CHECK-SAME:      [[MAP0]], [[MAP1]], [[MAP2]]
    // CHECK-SAME:      tensor<2x2x3xf32> into tensor<2x2x1x3xf32>

    // CHECK:       return [[VAL0]] : tensor<2x2x1x3xf32>
}
