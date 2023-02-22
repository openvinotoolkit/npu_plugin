//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-deconv-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertDeconv2DToConv2D
func @ConvertDeconv2DToConv2D(%arg0: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %FILTERS = const.Declare tensor<32x16x2x2xf16> = dense<1.000000e+00> : tensor<32x16x2x2xf16>
    %RESULT = IE.Deconvolution(%arg0, %FILTERS) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<32x16x2x2xf16> -> tensor<1x16x46x60xf16>
    return %RESULT : tensor<1x16x46x60xf16>

    // CHECK-NOT:   IE.Deconvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      pad_l = [1, 1, 0]
    // CHECK-SAME:      pad_r = [1, 1, 0]
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<32x16x2x2xf16
    // CHECK:       %[[CONV:.*]] = IE.Convolution([[UPS]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @ConvertDeconv2DToConv2DFQFilterFQParamsReused
func @ConvertDeconv2DToConv2DFQFilterFQParamsReused(%arg0: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %cst = const.Declare tensor<32x16x2x2xf16> = dense<1.000000e+00> : tensor<32x16x2x2xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %quantized_input = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<32x16x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x16x2x2xf16>

    %RESULT = IE.Deconvolution(%arg0, %quantized_input) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<32x16x2x2xf16> -> tensor<1x16x46x60xf16>
    return %RESULT : tensor<1x16x46x60xf16>

    // CHECK-NOT:   IE.Deconvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      pad_l = [1, 1, 0]
    // CHECK-SAME:      pad_r = [1, 1, 0]
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[REV_FILTER:%.*]] = const.Declare tensor<16x32x2x2xf16>
    // CHECK-SAME:      #const.Reverse<1 : i64>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize([[REV_FILTER]]
    // CHECK-SAME:      tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>
    // CHECK:       %[[CONV:.*]] = IE.Convolution([[UPS]], [[FQ]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @ConvertDeconv2DToConv2DFQFilterFQParamsUnique
func @ConvertDeconv2DToConv2DFQFilterFQParamsUnique(%arg0: tensor<1x32x23x30xf16>) -> tensor<1x16x46x60xf16> {
    %filter = const.Declare tensor<32x16x2x2xf16> = dense<1.000000e+00> : tensor<32x16x2x2xf16>
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<10.000000e+00> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<255.000000e+00> : tensor<1x1x1x1xf16>

    %quantized_input = IE.FakeQuantize(%filter, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = "NUMPY",
        levels = 256 : i64
    } : tensor<32x16x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x16x2x2xf16>

    %RESULT = IE.Deconvolution(%arg0, %quantized_input) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], output_padding = [0, 0]} : tensor<1x32x23x30xf16>, tensor<32x16x2x2xf16> -> tensor<1x16x46x60xf16>
    return %RESULT : tensor<1x16x46x60xf16>

    // CHECK-NOT:   IE.Deconvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      pad_l = [1, 1, 0]
    // CHECK-SAME:      pad_r = [1, 1, 0]
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[REV_FILTER:%.*]] = const.Declare tensor<16x32x2x2xf16>
    // CHECK-SAME:      #const.Reverse<1 : i64>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize([[REV_FILTER]]
    // CHECK-SAME:      tensor<16x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x32x2x2xf16>
    // CHECK:       %[[CONV:.*]] = IE.Convolution([[UPS]], [[FQ]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @ConvertDeconv2DToConv2DNonConstFilter
  func @ConvertDeconv2DToConv2DNonConstFilter(%arg0: tensor<1x16x30x30xf16>, %arg1: tensor<1x16x16x16xf16>) -> tensor<1x16x74x74xf16> {
     // deconv input
    %conv1_filter = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>

    %conv1 = IE.Convolution(%arg0, %conv1_filter) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1]
            } : tensor<1x16x30x30xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x30x30xf16>

    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<100.000000e+00> : tensor<1x1x1x1xf16>

    %fq_input = IE.FakeQuantize(%conv1, %cst_0, %cst_1, %cst_0, %cst_1) {
                auto_broadcast = "NUMPY",
                levels = 256 : i64
            } : tensor<1x16x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x30x30xf16>

    // deconv weights
    %conv2_filter = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>

    %conv2 = IE.Convolution(%arg1, %conv2_filter) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1]
            } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %fq_weights = IE.FakeQuantize(%conv2, %cst_2, %cst_3, %cst_2, %cst_3) {
                    auto_broadcast = "NUMPY",
                    levels = 256 : i64
                } : tensor<1x16x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x16x16xf16>

    //deconv
    %2 = IE.Deconvolution(%fq_input, %fq_weights) {
            dilations = [1, 1],
            output_padding = [0, 0],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2]
        } : tensor<1x16x30x30xf16>, tensor<1x16x16x16xf16> -> tensor<1x16x74x74xf16>

    return %2 : tensor<1x16x74x74xf16>

    // CHECK-NOT:   IE.Upsampling
    // CHECK:       [[CONV1:%.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x30x30xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x30x30xf16>
    // CHECK:       [[FQ1:%.*]] = IE.FakeQuantize([[CONV1]],
    // CHECK-SAME:      tensor<1x16x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x30x30xf16>
    // CHECK:       [[CONV2:%.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       [[FQ2:%.*]] = IE.FakeQuantize([[CONV2]],
    // CHECK-SAME:      tensor<1x16x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       %[[DECONV:.*]] = IE.Deconvolution([[FQ1]], [[FQ2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      output_padding = [0, 0],
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      tensor<1x16x30x30xf16>, tensor<1x16x16x16xf16> -> tensor<1x16x74x74xf16>
    // CHECK:       return %[[DECONV]]
}

// CHECK-LABEL: @ConvertDeconv2DToConv2DWhitOutputPadding
func @ConvertDeconv2DToConv2DWhitOutputPadding(%arg0: tensor<1x32x23x30xf16>) -> tensor<1x16x47x61xf16> {
    %FILTERS = const.Declare tensor<32x16x2x2xf16> = dense<1.000000e+00> : tensor<32x16x2x2xf16>
    %RESULT = IE.Deconvolution(%arg0, %FILTERS) {strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], output_padding = [1, 1]} : tensor<1x32x23x30xf16>, tensor<32x16x2x2xf16> -> tensor<1x16x47x61xf16>
    return %RESULT : tensor<1x16x47x61xf16>

    // CHECK-NOT:   IE.Deconvolution
    // CHECK:       [[UPS:%.*]] = IE.Upsampling
    // CHECK-SAME:      pad_l = [1, 1, 0]
    // CHECK-SAME:      pad_r = [1, 1, 0]
    // CHECK-SAME:      upsampling_factor = [2, 2, 1]
    // CHECK-SAME:      tensor<1x32x23x30xf16> -> tensor<1x32x47x61xf16>
    // CHECK:       [[WEIGHTS:%.*]] = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<32x16x2x2xf16
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[UPS]], [[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x32x47x61xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x46x60xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[CONV]] [0, 0, 45, 0] [1, 16, 1, 60] : tensor<1x16x46x60xf16> to tensor<1x16x1x60xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CONV]], [[SLICE1]]) {per_axis = {axis = 2 : i64}} : tensor<1x16x46x60xf16>, tensor<1x16x1x60xf16> -> tensor<1x16x47x60xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice [[CONCAT1]] [0, 0, 0, 59] [1, 16, 47, 1] : tensor<1x16x47x60xf16> to tensor<1x16x47x1xf16>
    // CHECK:       [[CONCAT2:%.*]] = IE.Concat([[CONCAT1]], [[SLICE2]]) {per_axis = {axis = 3 : i64}} : tensor<1x16x47x60xf16>, tensor<1x16x47x1xf16> -> tensor<1x16x47x61xf16>
    // CHECK:       return [[CONCAT2]] : tensor<1x16x47x61xf16>
}
