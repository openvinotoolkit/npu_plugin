// RUN: vpux-opt --split-input-file --lower-IE-to-quant %s | FileCheck %s

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

    // CHECK-NOT: IE.Constant

    // CHECK:       %[[VAL0:.*]] = "quant.const_fake_quant"(%arg0)
    // CHECK-SAME:      is_signed = false
    // CHECK-SAME:      max = 2.550000e+02
    // CHECK-SAME:      min = 0.000000e+00
    // CHECK-SAME:      narrow_range = false
    // CHECK-SAME:      num_bits = 8
    // CHECK:       return %[[VAL0]]
}

// -----

#low_vals = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>
#high_vals = dense<[101.0, 102.0, 103.0]> : tensor<3xf16>

// CHECK-LABEL: @PerAxis
func @PerAxis(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %input_low = IE.Constant tensor<1x3x1x1xf16> = #low_vals
    %input_high = IE.Constant tensor<1x3x1x1xf16> = #high_vals
    %output_low = IE.Constant tensor<1x3x1x1xf16> = #low_vals
    %output_high = IE.Constant tensor<1x3x1x1xf16> = #high_vals

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 255 : i32 } :
        tensor<1x3x30x30xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x30x30xf16>

    return %0 : tensor<1x3x30x30xf16>

    // CHECK-NOT: IE.Constant

    // CHECK:       %[[VAL0:.*]] = "quant.const_fake_quant_per_axis"(%arg0)
    // CHECK-SAME:      axis = 1
    // CHECK-SAME:      is_signed = false
    // CHECK-SAME:      max = [1.010000e+02 : f32, 1.020000e+02 : f32, 1.030000e+02 : f32]
    // CHECK-SAME:      min = [1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32]
    // CHECK-SAME:      narrow_range = true
    // CHECK-SAME:      num_bits = 8
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @UseDequantize
func @UseDequantize() -> tensor<1x3x30x30xf32> {
    %input_u8 = IE.Constant tensor<1x3x30x30xui8> = dense<10> : tensor<1x3x30x30xui8>
    %input_f32 = IE.Convert(%input_u8) { dstType = f32 } : tensor<1x3x30x30xui8> -> tensor<1x3x30x30xf32>

    %input_low = IE.Constant tensor<1x1x1x1xf32> = dense<0.0> : tensor<1xf32>
    %input_high = IE.Constant tensor<1x1x1x1xf32> = dense<255.0> : tensor<1xf32>
    %output_low = IE.Constant tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1xf32>
    %output_high = IE.Constant tensor<1x1x1x1xf32> = dense<10.0> : tensor<1xf32>

    %0 = IE.FakeQuantize(%input_f32, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Constant
    // CHECK-SAME:      tensor<1x3x30x30x!quant.uniform<u8:f32, 0.078431372549019607:128>>
    // CHECK-SAME:      = dense<10> : tensor<1x3x30x30xui8>
    // CHECK:       [[VAL1:%.*]] = "quant.dcast"([[VAL0]])
    // CHECK-SAME:      tensor<1x3x30x30x!quant.uniform<u8:f32, 0.078431372549019607:128>>
    // CHECK-SAME:      -> tensor<1x3x30x30xf32>
    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @UseDequantizeFP
func @UseDequantizeFP() -> tensor<1x3x30x30xf32> {
    %input = IE.Constant tensor<1x3x30x30xf32> = dense<10.0> : tensor<1x3x30x30xf32>

    %input_low = IE.Constant tensor<1x1x1x1xf32> = dense<0.0> : tensor<1xf32>
    %input_high = IE.Constant tensor<1x1x1x1xf32> = dense<255.0> : tensor<1xf32>
    %output_low = IE.Constant tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1xf32>
    %output_high = IE.Constant tensor<1x1x1x1xf32> = dense<10.0> : tensor<1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Constant
    // CHECK-SAME:      tensor<1x3x30x30x!quant.uniform<u8:f32, 0.078431372549019607:128>>
    // CHECK-SAME:      = dense<10> : tensor<1x3x30x30xui8>
    // CHECK:       [[VAL1:%.*]] = "quant.dcast"([[VAL0]])
    // CHECK-SAME:      tensor<1x3x30x30x!quant.uniform<u8:f32, 0.078431372549019607:128>>
    // CHECK-SAME:      -> tensor<1x3x30x30xf32>
    // CHECK:       return [[VAL1]]
}
