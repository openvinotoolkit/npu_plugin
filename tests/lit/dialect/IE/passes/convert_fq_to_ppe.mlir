// RUN: vpux-opt --split-input-file --convert-fq-to-ppe --canonicalize %s | FileCheck %s

!qElemType = type !quant.uniform<u8<0:254>:f32:0, {0.0078740157480314959:127,0.0086614175105658095:127,0.0094488192731001247:127,0.010236220096978615:127}>

// CHECK-LABEL: @ConvertToMaxPool
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x3x62x62xf32>) -> tensor<1x4x60x60xf32>
func @ConvertToMaxPool(%input: tensor<1x3x62x62xf32>) -> tensor<1x4x60x60xf32> {
    %input_low = const.Declare tensor<f32> = #const.Content<dense<0.0> : tensor<f32>>
    %input_high = const.Declare tensor<f32> = #const.Content<dense<255.0> : tensor<f32>>

    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights = const.Declare tensor<4x3x3x3xf32> = #const.Content<dense<128> : tensor<4x3x3x3xui8>, [#const.ConvertElemType<f32>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.Dequantize]>

    %conv = IE.Convolution(%input_fq, %weights)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    return %conv : tensor<1x4x60x60xf32>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<4x3x3x3xf32>

    // CHECK:       [[INPUT_FQ:%.*]] = IE.MaxPool([[INPUT]])
    // CHECK-SAME:      clip_op = {high = 2.550000e+02 : f64, low = 0.000000e+00 : f64
    // CHECK-SAME:      -> tensor<1x3x62x62xf32>

    // CHECK:       [[CONV:%.*]] = IE.Convolution([[INPUT_FQ]], [[WEIGHTS]])
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseWithConv
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x3x62x62xf32>) -> tensor<1x4x60x60xf32>
func @FuseWithConv(%input: tensor<1x3x62x62xf32>) -> tensor<1x4x60x60xf32> {
    %input_low = const.Declare tensor<f32> = #const.Content<dense<0.0> : tensor<f32>>
    %input_high = const.Declare tensor<f32> = #const.Content<dense<255.0> : tensor<f32>>
    %weights = const.Declare tensor<4x3x3x3xf32> = #const.Content<dense<128> : tensor<4x3x3x3xui8>, [#const.ConvertElemType<f32>]>

    %conv = IE.Convolution(%input, %weights)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    %out_fq = IE.FakeQuantize(%conv, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    return %out_fq : tensor<1x4x60x60xf32>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<4x3x3x3xf32>

    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, %0)
    // CHECK-SAME:      {clip_op = {high = 2.550000e+02 : f64, low = 0.000000e+00 : f64},
    // CHECK-SAME:      tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    // CHECK:       return [[CONV]]
}
