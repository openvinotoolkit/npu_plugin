// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @UseLeakyRelu
func @UseLeakyRelu(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %0 = IE.Constant tensor<1x16xf32> = dense<1.0> : tensor<16xf32>
    %1 = IE.PRelu(%arg0, %0) : 
        tensor<1x16x300x300xf32>, tensor<1x16xf32> -> tensor<1x16x300x300xf32>
    return %1 : tensor<1x16x300x300xf32>

    // CHECK:       %[[VAL0:.*]] = IE.LeakyRelu(%arg0) {negative_slope = 1.000000e+00 : f32} : tensor<1x16x300x300xf32> -> tensor<1x16x300x300xf32>
    // CHECK-NOT:   IE.PRelu
    // CHECK:       return %[[VAL0]]
}
