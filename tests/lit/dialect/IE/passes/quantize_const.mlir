// RUN: vpux-opt --split-input-file --quantize-const %s | FileCheck %s

!qtype = type !quant.uniform<u8:f32:0, {1.000000e-01:128, 2.000000e-01:128, 4.000000e-01:128, 8.000000e-01:128}>

// CHECK-LABEL: @PerAxis
func @PerAxis() -> tensor<4x1x1x1x!qtype> {
    %0 = IE.Constant tensor<4x1x1x1xf32> = dense<1.0> : tensor<4xf32>
    %1 = "quant.qcast"(%0) : (tensor<4x1x1x1xf32>) -> tensor<4x1x1x1x!qtype>
    return %1 : tensor<4x1x1x1x!qtype>

    // CHECK:       [[CST:%.*]] = IE.Constant
    // CHECK-SAME:      tensor<4x1x1x1x!quant.uniform<u8:f32:0, {1.000000e-01:128,2.000000e-01:128,4.000000e-01:128,8.000000e-01:128}>>
    // CHECK-SAME:      138
    // CHECK-SAME:      133
    // CHECK-SAME:      131
    // CHECK-SAME:      129

    // CHECK:       return [[CST]]
}
