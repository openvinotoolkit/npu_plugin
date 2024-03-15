//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-for-vpu %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @ConvertNonDepthWiseGroupTransposedConvToConv
func.func @ConvertNonDepthWiseGroupTransposedConvToConv(%arg0: tensor<1x32x64x64xf16>) -> tensor<1x32x128x128xf16> {
    %FILTERS = const.Declare tensor<2x16x16x4x4xf16> = dense<1.000000e+00> : tensor<2x16x16x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x64x64xf16>, tensor<2x16x16x4x4xf16> -> tensor<1x32x128x128xf16>
    return %RESULT : tensor<1x32x128x128xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x32x4x4xf16> = dense
    // CHECK:           [[UPSAMPLE:%.+]] = IE.Upsampling(%arg0)
    // CHECK-SAME:          {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [2, 2], pads_width = [2, 2]>, upsampling_factor = [2, 2, 1]} : tensor<1x32x64x64xf16> -> tensor<1x32x131x131xf16>
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[UPSAMPLE]], [[WEIGHTS]])
    // CHECK-SAME:          {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x131x131xf16>, tensor<32x32x4x4xf16> -> tensor<1x32x128x128xf16>

    // CHECK:           return [[CONV]]
}
