// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --fuse-post-ops %s | FileCheck %s

func @FakeQuantConv2dWithLeakyRelu1Test(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %cst = const.Declare tensor<f16> = #const.Content<dense<1.270000e+02> : tensor<f16>>
    %cst_0 = const.Declare tensor<f16> = #const.Content<dense<-1.280000e+02> : tensor<f16>>
    %cst_1 = const.Declare tensor<f16> = #const.Content<dense<6.000000e+00> : tensor<f16>>

    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x4x4xf16>

    %0 = IE.Convolution(%quantized_input, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %result = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x3x3xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x3x3xf16>

    return %result : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = {attrs = {negative_slope = 1.000000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.LeakyRelu
}

// -----

func @FakeQuantConv2dWithLeakyRelu15Test(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %cst = const.Declare tensor<f16> = #const.Content<dense<1.270000e+02> : tensor<f16>>
    %cst_0 = const.Declare tensor<f16> = #const.Content<dense<-1.280000e+02> : tensor<f16>>
    %cst_1 = const.Declare tensor<f16> = #const.Content<dense<6.000000e+00> : tensor<f16>>

    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x4x4xf16>

    %0 = IE.Convolution(%quantized_input, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %result = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<1x16x3x3xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x3x3xf16>

    return %result : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-NOT:      post_op = {attrs = {negative_slope = 1.500000e-01 : f64}, name = "IE.LeakyRelu"}
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NEXT:   IE.LeakyRelu
}
