// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

func @ConstFold() -> tensor<4x4xf32> {
    %0 = IE.Constant tensor<1x1x4x4xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.Constant tensor<2xsi64> = dense<[0, 1]> : tensor<2xsi64>
    %2 = IE.Squeeze(%0, %1) : tensor<1x1x4x4xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>

    // CHECK:       %[[VAL0:.*]] = IE.Constant tensor<4x4xf32> = dense<1.000000e+00> : tensor<16xf32>
    // CHECK-NOT:   IE.Constant
    // CHECK-NOT:   IE.Squeeze
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>

func @UseCollapsingReshape(%arg0 : tensor<1x16x1x20x1xf32>) -> tensor<16x20xf32> {
    %0 = IE.Constant tensor<3xsi64> = dense<[0, 2, 4]> : tensor<3xsi64>
    %1 = IE.Squeeze(%arg0, %0) : tensor<1x16x1x20x1xf32>, tensor<3xsi64> -> tensor<16x20xf32>
    return %1 : tensor<16x20xf32>

    // CHECK-NOT:   IE.Constant
    // CHECK-NOT:   IE.Squeeze

    // CHECK:       %[[VAL0:.*]] = linalg.tensor_reshape %arg0
    // CHECK-SAME:      [[MAP0]], [[MAP1]]
    // CHECK-SAME:      tensor<1x16x1x20x1xf32> into tensor<16x20xf32>

    // CHECK:       return %[[VAL0]] : tensor<16x20xf32>
}
