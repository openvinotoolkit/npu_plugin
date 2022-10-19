// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --low-precision %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8<0:254>:f32:0, {0.0078740157480314959:127,0.0086614175105658095:127,0.0094488192731001247:127,0.010236220096978615:127}>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @QuantizedConv
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x3x62x62xui8>) -> tensor<1x4x60x60xf32>
func @QuantizedConv(%input: tensor<1x3x62x62xui8>) -> tensor<1x4x60x60xf32> {
    %0 = IE.Convert(%input) {dstElemType = f32} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf32>

    %input_low = const.Declare tensor<f32> = #const.Content<dense<0.0> : tensor<f32>>
    %input_high = const.Declare tensor<f32> = #const.Content<dense<255.0> : tensor<f32>>

    %input_fq = IE.FakeQuantize(%0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x3x62x62xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x62x62xf32>

    %weights = const.Declare tensor<4x3x3x3xf32> = #const.Content<dense<128> : tensor<4x3x3x3xui8>, [#const.ConvertElemType<f32>]>

    %weights_in_low = const.Declare tensor<1xf32> = #const.Content<dense<0.0> : tensor<1xf32>>
    %weights_in_high = const.Declare tensor<1xf32> = #const.Content<dense<255.0> : tensor<1xf32>>

    %weights_out_low = const.Declare tensor<4x1x1x1xf32> = #const.Content<dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]], [[[-1.3]]]]> : tensor<4x1x1x1xf32>>
    %weights_out_high = const.Declare tensor<4x1x1x1xf32> = #const.Content<dense<[[[[1.0]]], [[[1.1]]], [[[1.2]]], [[[1.3]]]]> : tensor<4x1x1x1xf32>>

    %weights_fq = IE.FakeQuantize(%weights, %weights_in_low, %weights_in_high, %weights_out_low, %weights_out_high)
        { auto_broadcast = "NUMPY", levels = 255 } :
        tensor<4x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x3x3x3xf32>

    %conv = IE.Convolution(%input_fq, %weights_fq)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x3x62x62xf32>, tensor<4x3x3x3xf32> -> tensor<1x4x60x60xf32>

    %last_fq = IE.FakeQuantize(%conv, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x4x60x60xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x4x60x60xf32>

    return %last_fq : tensor<1x4x60x60xf32>

    // CHECK:     [[WEIGHTS:%.*]] = const.Declare
    // CHECK-SAME:    #const.Content<dense<128> : tensor<4x3x3x3xui8>,
    // CHECK-SAME:    [#const.ConvertElemType<f32>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>

    // CHECK:     [[INPUT_QUANT:%.*]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:     tensor<1x3x62x62xui8> -> tensor<1x3x62x62x!qElemType1>

    // CHECK:     [[CONV:%.*]] = IE.Convolution([[INPUT_QUANT]], [[WEIGHTS]])

    // CHECK:     [[OUT_DEQ:%.*]] = IE.And([[CONV]], [[CONV]]) {auto_broadcast = "NONE_OR_EXPLICIT"}
    // CHECK-SAME:     -> tensor<1x4x60x60xf32>

    // CHECK:     return [[OUT_DEQ]]
}
