// RUN: vpux-opt --split-input-file --low-precision %s | FileCheck %s

// CHECK-LABEL: @QuantizedConv
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x3x62x62xf32>) -> tensor<1x4x60x60xf32>
func @QuantizedConv(%input: tensor<1x3x62x62xf32>) -> tensor<1x4x60x60xf32> {
    %input_low = IE.Constant tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = IE.Constant tensor<f32> = dense<255.0> : tensor<f32>

    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights = IE.Constant tensor<4x3x3x3xf32> = dense<128.0> : tensor<4x3x3x3xf32>

    %weights_in_low = IE.Constant tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %weights_in_high = IE.Constant tensor<1xf32> = dense<255.0> : tensor<1xf32>

    %weights_out_low = IE.Constant tensor<4x1x1x1xf32> = dense<[-1.0, -1.1, -1.2, -1.3]> : tensor<4xf32>
    %weights_out_high = IE.Constant tensor<4x1x1x1xf32> = dense<[1.0, 1.1, 1.2, 1.3]> : tensor<4xf32>

    %weights_fq = IE.FakeQuantize(%weights, %weights_in_low, %weights_in_high, %weights_out_low, %weights_out_high)
        { auto_broadcast = "NUMPY", levels = 255 : i32 } :
        tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x3x3x3xf32>

    %conv = IE.Convolution(%input_fq, %weights_fq)
        {
            strides = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            dilations = [1 : i32, 1 : i32]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    return %conv : tensor<1x4x60x60xf32>

    // CHECK:       [[WEIGHTS_FQ:%.*]] = IE.Constant
    // CHECK-SAME:      tensor<4x3x3x3xf32>
    // CHECK-SAME:      = dense<0.000000e+00> : tensor<4x3x3x3xf32>

    // CHECK:       [[INPUT_MIN:%.*]] = IE.Constant tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK:       [[INPUT_MAX:%.*]] = IE.Constant tensor<f32> = dense<2.550000e+02> : tensor<f32>

    // CHECK:       [[INPUT_FQ:%.*]] = IE.FakeQuantize([[INPUT]], [[INPUT_MIN]], [[INPUT_MAX]], [[INPUT_MIN]], [[INPUT_MAX]])
    // CHECK-SAME:      levels = 256

    // CHECK:       [[CONV:%.*]] = IE.Convolution([[INPUT_FQ]], [[WEIGHTS_FQ]])
    // CHECK:       return [[CONV]]
}
