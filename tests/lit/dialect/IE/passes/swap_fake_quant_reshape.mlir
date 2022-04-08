// RUN: vpux-opt --split-input-file --swap-fake-quant-reshape %s | FileCheck %s

// CHECK-LABEL: @SwapFakeQuantReshape
func @SwapFakeQuantReshape(
        %input: tensor<1x1x40xf16>,
        %weights: tensor<512x40x1x1xf16>)
            -> tensor<1x512x1x1xf16> {
    %cst_0 = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f16>>
    %cst_1 = const.Declare tensor<f16> = #const.Content<dense<1.000000e+00> : tensor<f16>>
    %1 = IE.SoftMax(%input) {axisInd = 2} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    %2 = IE.AffineReshape(%1) {shape_value = [1, 1, 1, 40], dim_mapping = [[0], [1, 2], [3]]} : tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    %3 = IE.FakeQuantize(%2, %cst_0, %cst_1, %cst_0, %cst_1) {
                     auto_broadcast = "NUMPY",
                     levels = 256 : i64
                 } : tensor<1x1x1x40xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x40xf16>
    %4 = IE.AffineReshape(%3) {shape_value = [1, 40, 1, 1], dim_mapping = [[0], [0], [0], [1, 2, 3]]} : tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    %5 = IE.Convolution(%4, %weights) {
                    strides = [1, 1],
                    pads_begin = [0, 0],
                    pads_end = [0, 0],
                    dilations = [1, 1]
                } : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>

    return %5 : tensor<1x512x1x1xf16>

    // CHECK:       %[[FQ_MAX:.*]] = const.Declare tensor<f16> = #const.Content<dense<1.000000e+00> : tensor<f16>>
    // CHECK:       %[[FQ_MIN:.*]] = const.Declare tensor<f16> = #const.Content<dense<0.000000e+00> : tensor<f16>>
    // CHECK:       %[[SOFTMAX:.*]] = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x1x40xf16> -> tensor<1x1x40xf16>
    // CHECK:       %[[AFFINERESHAPE1:.*]] = IE.AffineReshape(%[[SOFTMAX]])
    // CHECK-SAME:      tensor<1x1x40xf16> -> tensor<1x1x1x40xf16>
    // CHECK:       %[[AFFINERESHAPE2:.*]] = IE.AffineReshape(%[[AFFINERESHAPE1]])
    // CHECK-SAME:      tensor<1x1x1x40xf16> -> tensor<1x40x1x1xf16>
    // CHECK:       %[[FQ:.*]] = IE.FakeQuantize(%[[AFFINERESHAPE2]], %[[FQ_MIN]], %[[FQ_MAX]], %[[FQ_MIN]], %[[FQ_MAX]])
    // CHECK-SAME:      {auto_broadcast = "NUMPY", levels = 256 : i64}
    // CHECK-SAME:      tensor<1x40x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16>
    // CHECK-SAME:         -> tensor<1x40x1x1xf16>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[FQ]], %arg1)
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>
}
