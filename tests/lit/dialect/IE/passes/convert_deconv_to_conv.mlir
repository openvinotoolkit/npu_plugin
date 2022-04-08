// RUN: vpux-opt --split-input-file --convert-deconv-to-conv %s | FileCheck %s

// CHECK-LABEL: @ConvertDeconv2DToConv2D
func @ConvertDeconv2DToConv2D(%arg0: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %FILTERS = const.Declare tensor<32x16x2x2xf16> = #const.Content<dense<1.000000e+00> : tensor<32x16x2x2xf16>>
    %RESULT = IE.Deconvolution(%arg0, %FILTERS) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<32x16x2x2xf16> -> tensor<1x16x46x60xf16>
    return %RESULT : tensor<1x16x46x60xf16>

    // CHECK-NOT:   IE.Deconvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      pad_l = [1, 1, 0]
    // CHECK-SAME:      pad_r = [1, 1, 0]
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16> 
    // CHECK:       [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16> = #const.Content<dense<1.000000e+00> : tensor<32x16x2x2xf16>
    // CHECK:       %[[CONV:.*]] = IE.Convolution([[UPS]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return %[[CONV]]
}
