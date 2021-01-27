// RUN: vpux-opt --split-input-file --dequantize-const %s | FileCheck %s

// CHECK-LABEL: @PerAxis
func @PerAxis() -> tensor<4x1x1x1xf32> {
    %0 = IE.Constant
        tensor<4x1x1x1x!quant.uniform<u8:f32:0, {0.1:128, 0.2:128, 0.3:128, 0.4:128}>> =
            dense<129> : tensor<4x1x1x1xui8>

    %1 = "quant.dcast"(%0) :
        (tensor<4x1x1x1x!quant.uniform<u8:f32:0, {0.1:128, 0.2:128, 0.3:128, 0.4:128}>>)
            -> tensor<4x1x1x1xf32>

    return %1 : tensor<4x1x1x1xf32>

    // CHECK:       [[CST:%.*]] = IE.Constant
    // CHECK-SAME:      tensor<4x1x1x1xf32>
    // CHECK-SAME:      1.000000e-01
    // CHECK-SAME:      2.000000e-01
    // CHECK-SAME:      3.000000e-01
    // CHECK-SAME:      4.000000e-01

    // CHECK:       return [[CST]]
}
