// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --resolve-pwl-post-ops %s | FileCheck %s

// CHECK-LABEL: @UnfuseConvSigmoid
func @UnfuseConvSigmoid(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %cst)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = {attrs = {}, name = "IE.Sigmoid"},
            strides = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    return %0 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     strides = [1, 1]
    // CHECK:   IE.Sigmoid
}

// -----

!qElemType0 = type !quant.uniform<ui8:f32, 1.000000e+00>
!qElemType1 = type !quant.uniform<ui8:f32, 9.8455479662789983E-4>
!qElemType2 = type !quant.uniform<ui8:f32, 0.0040160642570281121:3>

// CHECK-LABEL: @InsertQuantizeCastConvSigmoid
func @InsertQuantizeCastConvSigmoid(%arg0: tensor<1x16x4x4x!qElemType0>) -> tensor<1x16x3x3xf32> {
    %cst = const.Declare tensor<16x16x2x2x!qElemType0> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>
    %0 = IE.Convolution(%arg0, %cst)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = {attrs = {}, name = "IE.Sigmoid"},
            strides = [1, 1]
        } :
        tensor<1x16x4x4x!qElemType0>, tensor<16x16x2x2x!qElemType0> -> tensor<1x16x3x3x!qElemType0>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x16x3x3x!qElemType0> -> tensor<1x16x3x3xf32>
    return %1 : tensor<1x16x3x3xf32>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {}, name = "IE.Sigmoid"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-SAME:     tensor<1x16x4x4x!qElemType0>, tensor<16x16x2x2x!qElemType0> ->
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType1>

    // CHECK:       [[VAL0:%.*]] = IE.QuantizeCast
    // CHECK-SAME:     {dstElemType = !qElemType2}
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType1> ->
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType2>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:     {dstElemType = f32}
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType2> -> tensor<1x16x3x3xf32>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.0039215686274509803>
!qElemType1 = type !quant.uniform<u8:f16, 0.023529411764705882:128>
!qElemType2 = type !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @NoInsertConvLeakyRelu
func @NoInsertConvLeakyRelu(%arg0: tensor<1x16x4x4x!qElemType1>) -> tensor<1x16x3x3xf32> {
    %cst = const.Declare tensor<16x16x2x2x!qElemType0> = #const.Content<dense<1.0> : tensor<16x16x2x2xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>
    %0 = IE.Convolution(%arg0, %cst)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = {attrs = {negative_slope = 0.10000000149011612 : f64}, name = "IE.LeakyRelu"},
            strides = [1, 1]
        } :
        tensor<1x16x4x4x!qElemType1>, tensor<16x16x2x2x!qElemType0> -> tensor<1x16x3x3x!qElemType2>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x16x3x3x!qElemType2> -> tensor<1x16x3x3xf32>
    return %1 : tensor<1x16x3x3xf32>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 0.10000000149011612 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-SAME:     tensor<1x16x4x4x!qElemType0>, tensor<16x16x2x2x!qElemType1> ->
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType2>
    // CHECK-NOT:    IE.QuantizeCast
    // CHECK-NOT:    IE.Quantize

    // CHECK:       [[VAL0:%.*]] = IE.Dequantize
    // CHECK-SAME:     {dstElemType = f32}
    // CHECK-SAME:     tensor<1x16x3x3x!qElemType2> -> tensor<1x16x3x3xf32>

    // CHECK:       return [[VAL0]]
}
