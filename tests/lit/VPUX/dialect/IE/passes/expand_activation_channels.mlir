//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandMaxPoolChannels
func @ExpandMaxPoolChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x13xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = "FLOOR", strides = [2, 2]
    } : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    return %0 : tensor<1x3x15x13xf16, {order = #NHWC}>
}

// CHECK:       [[PAD:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[POOL:%.*]] = IE.MaxPool([[PAD]])
// CHECK-SAME:      kernel_size = [5, 5]
// CHECK-SAME:      pads_begin = [2, 0]
// CHECK-SAME:      pads_end = [2, 0]
// CHECK-SAME:      rounding_type = "FLOOR"
// CHECK-SAME:      strides = [2, 2]
// CHECK-SAME:      -> tensor<1x16x15x13xf16, {order = #NHWC}>

// CHECK:       [[OUT:%.*]] = IE.Slice [[POOL]] [0, 0, 0, 0] [1, 3, 15, 13]

// CHECK        return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

// CHECK-LABEL: @ExpandQuantMaxPoolChannels
func @ExpandQuantMaxPoolChannels(%input: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>) -> tensor<1x3x15x13x!qElemType1, {order = #NHWC}> {
    %1 = IE.MaxPool(%input) {
        kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = "FLOOR", strides = [2, 2]
    } : tensor<1x3x30x30x!qElemType0, {order = #NHWC}> -> tensor<1x3x15x13x!qElemType1, {order = #NHWC}>
    return %1 : tensor<1x3x15x13x!qElemType1, {order = #NHWC}>
}

// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[MAXPOOL_OUT:%.+]] = IE.MaxPool([[EXPAND_OUT]])
// CHECK-SAME:      kernel_size = [5, 5]
// CHECK-SAME:      pads_begin = [2, 0]
// CHECK-SAME:      pads_end = [2, 0]
// CHECK-SAME:      rounding_type = "FLOOR"
// CHECK-SAME:      strides = [2, 2]
// CHECK-SAME:      -> tensor<1x16x15x13x!qElemType2, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[MAXPOOL_OUT]] [0, 0, 0, 0] [1, 3, 15, 13]

// CHECK:       return [[SLICE_OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwiseAddChannels
func @ExpandEltwiseAddChannels(%arg0: tensor<1x3x30x25xf16, {order = #NHWC}>, %arg1: tensor<1x3x30x25xf16, {order = #NHWC}>)
        -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} :
        tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %0 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK:       [[EXPAND_LEFT_INPUT:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[EXPAND_RIGHT_INPUT:%.*]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[ELTWISE_ADD:%.*]] = IE.Add([[EXPAND_LEFT_INPUT]], [[EXPAND_RIGHT_INPUT]])
// CHECK:       [[OUT:%.*]] = IE.Slice [[ELTWISE_ADD]] [0, 0, 0, 0] [1, 3, 30, 25]
// CHECK        return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvolutionChannels
func @ExpandGroupConvolutionChannels(%arg0: tensor<1x72x56x56xf16, {order = #NHWC}>) -> tensor<1x72x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<72x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<72x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = const.Declare tensor<1x72x1x1xf16> = dense<1.0> : tensor<1x72x1x1xf16>

    %2 = IE.GroupConvolution(%arg0, %0, %1) {
        dilations = [1, 1], groups = 72, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x72x56x56xf16, {order = #NHWC}>, tensor<72x1x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x72x28x28xf16, {order = #NHWC}>
}

// CHECK:       [[EXTENDED_GROUP:%.*]] = const.Declare tensor<1x80x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
// CHECK:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 0, 0, 0]>]

// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.GroupConvolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_GROUP]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]

// CHECK        return [[REDUNDRANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannels
func @ExpandConvolutionChannels(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>, %arg1: tensor<40x77x1x1xf16, {order = #NHWC}>) -> tensor<1x40x56x56xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56xf16, {order = #NHWC}>, tensor<40x77x1x1xf16, {order = #NHWC}> -> tensor<1x40x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x40x56x56xf16, {order = #NHWC}>
}

// CHECK:       [[CST:%.*]] = const.Declare tensor<40x3x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<40x3x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[CONCAT:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<40x77x1x1xf16, {order = #NHWC}>, tensor<40x3x1x1xf16, {order = #NHWC}> -> tensor<40x80x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_FILTER:%.*]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<40x80x1x1xf16, {order = #NHWC}> -> tensor<48x80x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]
// CHECK        return [[REDUNDRANT_SUBTENSOR]]


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsOnlyIC
func @ExpandConvolutionChannelsOnlyIC(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>, %arg1: tensor<48x77x1x1xf16, {order = #NHWC}>) -> tensor<1x48x56x56xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56xf16, {order = #NHWC}>, tensor<48x77x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x48x56x56xf16, {order = #NHWC}>
}

// CHECK:       [[CST:%.*]] = const.Declare tensor<48x3x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<48x3x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_FILTER:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<48x77x1x1xf16, {order = #NHWC}>, tensor<48x3x1x1xf16, {order = #NHWC}> -> tensor<48x80x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK        return [[EXTENDED_CONV]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsConst
func @ExpandConvolutionChannelsConst(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>) -> tensor<1x40x56x56xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<40x77x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<40x77x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %filter)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56xf16, {order = #NHWC}>, tensor<40x77x1x1xf16, {order = #NHWC}> -> tensor<1x40x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x40x56x56xf16, {order = #NHWC}>
}

// CHECK:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<48x80x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 3, 0, 0]>]
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]
// CHECK        return [[REDUNDRANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844:127>

// CHECK-LABEL: @ExpandConvolutionChannelsOnlyICWithQuant
func @ExpandConvolutionChannelsOnlyICWithQuant(%arg0: tensor<1x77x56x56x!qElemType0, {order = #NHWC}>, %arg1: tensor<48x77x1x1x!qElemType0, {order = #NHWC}>) -> tensor<1x48x56x56x!qElemType0, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56x!qElemType0, {order = #NHWC}>, tensor<48x77x1x1x!qElemType0, {order = #NHWC}> -> tensor<1x48x56x56x!qElemType0, {order = #NHWC}>

    return %0 : tensor<1x48x56x56x!qElemType0, {order = #NHWC}>
}

// CHECK:       [[CST:%.*]] = const.Declare tensor<48x3x1x1x!qElemType, {order = #NHWC}> = dense<127> : tensor<48x3x1x1xui8, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_FILTER:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<48x77x1x1x!qElemType, {order = #NHWC}>, tensor<48x3x1x1x!qElemType, {order = #NHWC}> -> tensor<48x80x1x1x!qElemType, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK        return [[EXTENDED_CONV]]


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127}>
!qElemType3 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

func @ExpandQuantGroupConvolutionChannels(
        %input: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>,
        %filter: tensor<3x1x3x3x!qElemType1, {order = #NHWC}>)
            -> tensor<1x3x15x15x!qElemType2, {order = #NHWC}> {
    %1 = IE.GroupConvolution(%input, %filter) {
        dilations = [1, 1], groups = 3, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x3x30x30x!qElemType0, {order = #NHWC}>, tensor<3x1x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x3x15x15x!qElemType2, {order = #NHWC}>
    return %1 : tensor<1x3x15x15x!qElemType2, {order = #NHWC}>
}

// CHECK-LABEL: func @ExpandQuantGroupConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<3x1x3x3x!qElemType1, {order = #NHWC}>

// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[PAD_OUT:%.+]] = IE.Expand([[FILTER]]) {pads_begin = [0, 0, 0, 0], pads_end = [13, 0, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.GroupConvolution([[EXPAND_OUT]], [[PAD_OUT]])
// CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
// CHECK-SAME:      -> tensor<1x16x15x15x!qElemType4, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 3, 15, 15]

// CHECK:       return [[SLICE_OUT]] : tensor<1x3x15x15x!qElemType2, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ExpandQuantGroupConvolutionChannelsConst(%input: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x15xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<3x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.GroupConvolution(%input, %filter) {
        dilations = [1, 1], groups = 3, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<3x1x3x3xf16, {order = #NHWC}> -> tensor<1x3x15x15xf16, {order = #NHWC}>
    return %1 : tensor<1x3x15x15xf16, {order = #NHWC}>
}

// CHECK-LABEL: func @ExpandQuantGroupConvolutionChannelsConst
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30xf16, {order = #NHWC}>

// CHECK:       [[EXPAND_FILTER:%.*]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>]
// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.GroupConvolution([[EXPAND_OUT]], [[EXPAND_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
// CHECK-SAME:      -> tensor<1x16x15x15xf16, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 3, 15, 15]

// CHECK:       return [[SLICE_OUT]]
