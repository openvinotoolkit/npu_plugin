//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX num-of-dpu-groups=1" --expand-activation-channels --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoChannelExpandAndOp
func.func @NoChannelExpandAndOp(%arg0: tensor<1x3x30x25xf16, {order = #NHWC}>) -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %0 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK-NOT:   IE.Expand
// CHECK:       [[SW_AND:%.*]] = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-NOT:   IE.Slice
// CHECK:       return [[SW_AND]] : tensor<1x3x30x25xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandZMajorConvChannelsFP16
func.func @ExpandZMajorConvChannelsFP16(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      dense<1.000000e+00> : tensor<5x3x3x3xf16>,
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>

// CHECK:       [[REDUNDANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
// CHECK        return [[REDUNDANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8<0:254>:f16:0, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
!qElemType1 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType2 = !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @ExpandZMajorConvChannelsQuant
func.func @ExpandZMajorConvChannelsQuant(%arg0: tensor<1x3x30x30x!qElemType1, {order = #NHWC}>) -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3x!qElemType0, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16>, [
            #const.Reorder<#NHWC>,
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType0>
    ]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType1, {order = #NHWC}>, tensor<5x3x3x3x!qElemType0, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}>

    return %1 : tensor<1x5x28x28x!qElemType2, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x4x3x3x!qElemType2, {order = #NHWC}> =
// CHECK-SAME:      dense<1.000000e+00> : tensor<5x3x3x3xf16>,
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType3>, #const.PadWithZero<[0, 0, 0, 0], [11, 1, 0, 0]>]
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x30x30x!qElemType0, {order = #NHWC}> -> tensor<1x4x30x30x!qElemType0, {order = #NHWC}>

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK-SAME:      -> tensor<1x16x28x28x!qElemType1, {order = #NHWC}>

// CHECK:       [[REDUNDANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
// CHECK        return [[REDUNDANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

func.func @ExpandQuantConvolutionChannels(%input: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>)
            -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3x!qElemType2, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16, {order = #NHWC}>, [
        #const.ConvertElemType<ui8>,
        #const.QuantCast<!qElemType2>
    ]
    %1 = IE.Convolution(%input, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType0, {order = #NHWC}>, tensor<5x3x3x3x!qElemType2, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}>
    return %1 : tensor<1x5x28x28x!qElemType1, {order = #NHWC}>
}

// CHECK-LABEL: func.func @ExpandQuantConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>

// CHECK-DAG:       [[PADDED_FILTER:%.*]] = const.Declare tensor<16x4x3x3x!qElemType2, {order = #NHWC}> =
// CHECK-SAME:      dense<1.000000e+00> : tensor<5x3x3x3xf16, {order = #NHWC}>, [
// CHECK-SAME:          #const.ConvertElemType<ui8>,
// CHECK-SAME:          #const.QuantCast<!qElemType3>,
// CHECK-SAME:          #const.PadWithZero<[0, 0, 0, 0], [11, 1, 0, 0]>
// CHECK-SAME:      ]

// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.Convolution([[EXPAND_OUT]], [[PADDED_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
// CHECK-SAME:      -> tensor<1x16x28x28x!qElemType4, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 5, 28, 28] :

// CHECK:       return [[SLICE_OUT]] : tensor<1x5x28x28x!qElemType1, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandBiasesConvolutionChannels
func.func @ExpandBiasesConvolutionChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = const.Declare tensor<1x5x1x1xf16> = dense<1.0> : tensor<1x5x1x1xf16>

    %2 = IE.Convolution(%arg0, %0, %1) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}>, tensor<1x5x1x1xf16> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]

// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_BIAS]])
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
// CHECK:       [[REDUNDANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
// CHECK-SAME:       : tensor<1x16x28x28xf16, {order = #NHWC}> to tensor<1x5x28x28xf16, {order = #NHWC}>
// CHECK        return [[REDUNDANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ExpandConvolutionChannelsWithAdd(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x3x30x30xf16, {order = #NHWC}>
        -> tensor<1x3x30x30xf16, {order = #NHWC}>
    %1 = IE.Convolution(%0, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK-LABEL: func.func @ExpandConvolutionChannelsWithAdd

// CHECK-DAG:       [[EXPAND_FILTER:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]
// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[VAR0:%.*]] = IE.Add([[EXPAND_OUT]], [[EXPAND_OUT]])
// CHECK-SAME:       tensor<1x16x30x30xf16, {order = #NHWC}>, tensor<1x16x30x30xf16, {order = #NHWC}> 
// CHECK-SAME:       -> tensor<1x16x30x30xf16, {order = #NHWC}>

// CHECK:       [[CONV_OUT:%.+]] = IE.Convolution([[VAR0]], [[EXPAND_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 5, 28, 28]

// CHECK:       return [[SLICE_OUT]] : tensor<1x5x28x28xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<u8:f16, 0.25>

func.func @ExpandConvolutionChannelsWithAddAndQuantCast(%arg0: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>) 
        -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3x!qElemType0, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16, {order = #NHWC}>, [
        #const.ConvertElemType<ui8>,
        #const.QuantCast<!qElemType0>
    ]
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30x!qElemType0, {order = #NHWC}>, tensor<1x3x30x30x!qElemType0, {order = #NHWC}>
        -> tensor<1x3x30x30x!qElemType0, {order = #NHWC}>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : 
        tensor<1x3x30x30x!qElemType0, {order = #NHWC}> -> tensor<1x3x30x30x!qElemType1, {order = #NHWC}>
    %2 = IE.Convolution(%1, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType1, {order = #NHWC}>, tensor<5x3x3x3x!qElemType0, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}>
    return %2 : tensor<1x5x28x28x!qElemType1, {order = #NHWC}>
}

// CHECK-LABEL: func.func @ExpandConvolutionChannelsWithAddAndQuantCast
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>

// CHECK-DAG:       [[EXPAND_FILTER:%.*]] = const.Declare tensor<16x4x3x3x!qElemType0, {order = #NHWC}>
// CHECK-SAME:      [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>, #const.PadWithZero<[0, 0, 0, 0], [11, 1, 0, 0]>]
// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[VAR0:%.*]] = IE.Add([[EXPAND_OUT]]
// CHECK:       [[SLICE_ADD:%.+]] = IE.Slice [[VAR0]] [0, 0, 0, 0] [1, 3, 30, 30]
// CHECK:       [[QUANTCAST:%.*]] = IE.QuantizeCast([[SLICE_ADD]]
// CHECK:       [[EXPAND_ADD:%.+]] = IE.Expand([[QUANTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.Convolution([[EXPAND_ADD]], [[EXPAND_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
// CHECK-SAME:      -> tensor<1x16x28x28x!qElemType1, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 5, 28, 28]

// CHECK:       return [[SLICE_OUT]] : tensor<1x5x28x28x!qElemType1, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandZMajorConvChannelsOnlyWeightFP16
func.func @ExpandZMajorConvChannelsOnlyWeightFP16(%arg0: tensor<1x4x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x4x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<5x4x3x3xf16>, [#const.Reorder<#NHWC>]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x4x30x30xf16, {order = #NHWC}>, tensor<5x4x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x4x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      dense<1.000000e+00> : tensor<5x4x3x3xf16>,
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 0, 0, 0]>]

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution(%arg0, [[EXTENDED_FILTER]])
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>

// CHECK:       [[REDUNDANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
// CHECK        return [[REDUNDANT_SUBTENSOR]]
