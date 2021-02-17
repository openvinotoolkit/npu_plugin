// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FuseMulAndAdd
func @FuseMulAndAdd(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = IE.Constant tensor<1x3x1x1xf32> = dense<1.0> : tensor<1x3x1x1xf32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias = IE.Constant tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %1 = IE.Add(%0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK:       %[[WEIGHTS:.*]] = IE.Constant tensor<1x3x1x1xf32> = dense<1.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[BIAS:.*]] = IE.Constant tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK:       return %[[VAL0]]
}

