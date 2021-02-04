// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ConstFold
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
