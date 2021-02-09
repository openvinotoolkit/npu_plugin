// RUN: vpux-opt --split-input-file --split-fake-quant %s | FileCheck %s

// CHECK-LABEL: @SingleQuantParams
func @SingleQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = IE.Constant tensor<1x1x1x1xf32> = dense<0.0> : tensor<1xf32>
    %input_high = IE.Constant tensor<1x1x1x1xf32> = dense<255.0> : tensor<1xf32>
    %output_low = IE.Constant tensor<1x1x1x1xf32> = dense<0.0> : tensor<1xf32>
    %output_high = IE.Constant tensor<1x1x1x1xf32> = dense<255.0> : tensor<1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = "quant.qcast"(%arg0)
    // CHECK-SAME:      (tensor<1x3x30x30xf32>) ->
    // CHECK-SAME:      tensor<1x3x30x30x!quant.uniform<u8:f32, 1.000000e+00>>

    // CHECK:       [[VAL1:%.*]] = "quant.dcast"([[VAL0]])
    // CHECK-SAME:      (tensor<1x3x30x30x!quant.uniform<u8:f32, 1.000000e+00>>) ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @UseDequantize
func @UseDequantize() -> tensor<1x3x30x30xf32> {
    %input = IE.Constant tensor<1x3x30x30xf32> = dense<10.0> : tensor<1x3x30x30xf32>

    %input_low = IE.Constant tensor<1x1x1x1xf32> = dense<0.0> : tensor<1xf32>
    %input_high = IE.Constant tensor<1x1x1x1xf32> = dense<255.0> : tensor<1xf32>
    %output_low = IE.Constant tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1xf32>
    %output_high = IE.Constant tensor<1x1x1x1xf32> = dense<10.0> : tensor<1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Constant tensor<1x3x30x30x!quant.uniform<u8:f32, 0.078431372549019607:128>>
    // CHECK-SAME:      = dense<1.000000e+01> : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL1:%.*]] = "quant.dcast"([[VAL0]])
    // CHECK-SAME:      tensor<1x3x30x30x!quant.uniform<u8:f32, 0.078431372549019607:128>>
    // CHECK-SAME:      -> tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}
