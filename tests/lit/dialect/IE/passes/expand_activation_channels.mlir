// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --expand-activation-channels --canonicalize %s | FileCheck %s

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

// CHECK-LABEL: @ExpandZMajorConvChannels
func @ExpandZMajorConvChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> =
        #const.Content<dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]>

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<5x3x3x3xf16>,
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>

// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
// CHECK        return [[REDUNDRANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandCMajorConvChannels
func @ExpandCMajorConvChannels(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x5x32x32xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x1x1xf16, {order = #NHWC}> =
        #const.Content<dense<1.0> : tensor<5x3x1x1xf16>, [#const.Reorder<#NHWC>]>

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x32x32xf16>, tensor<5x3x1x1xf16, {order = #NHWC}> -> tensor<1x5x32x32xf16, {order = #NHWC}>

    return %1 : tensor<1x5x32x32xf16, {order = #NHWC}>
}

// CHECK:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}> =
// CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<5x3x1x1xf16>,
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 0, 0, 0]>]>

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution(%arg0, [[EXTENDED_FILTER]])
// CHECK-SAME:      -> tensor<1x16x32x32xf16, {order = #NHWC}>

// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 5, 32, 32]
// CHECK        return [[REDUNDRANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
!qElemType3 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

func @ExpandQuantConvolutionChannels(
        %input: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>,
        %filter: tensor<5x3x3x3x!qElemType1, {order = #NHWC}>)
            -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}> {
    %1 = IE.Convolution(%input, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType0, {order = #NHWC}>, tensor<5x3x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}>
    return %1 : tensor<1x5x28x28x!qElemType2, {order = #NHWC}>
}

// CHECK-LABEL: func @ExpandQuantConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType0, {order = #NHWC}>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<5x3x3x3x!qElemType1, {order = #NHWC}>

// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[PAD_OUT:%.+]] = IE.Expand([[FILTER]]) {pads_begin = [0, 0, 0, 0], pads_end = [11, 13, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.Convolution([[EXPAND_OUT]], [[PAD_OUT]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
// CHECK-SAME:      -> tensor<1x16x28x28x!qElemType4, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 5, 28, 28] :

// CHECK:       return [[SLICE_OUT]] : tensor<1x5x28x28x!qElemType2, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandBiasesConvolutionChannels
func @ExpandBiasesConvolutionChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]>
    %1 = const.Declare tensor<1x5x1x1xf16> = #const.Content<dense<1.0> : tensor<1x5x1x1xf16>>

    %2 = IE.Convolution(%arg0, %0, %1) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}>, tensor<1x5x1x1xf16> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK:       [[EXTENDED_BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]>
// CHECK:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>

// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_BIAS]])
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]
// CHECK        return [[REDUNDRANT_SUBTENSOR]]

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

// CHECK-LABEL: @ExpandEltwiseAndChannelsSameInputs
func @ExpandEltwiseAndChannelsSameInputs(%arg0: tensor<1x3x30x25xf16, {order = #NHWC}>) -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = "NUMPY"} :
            tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %0 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK:       [[EXPAND_INPUT:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[ELTWISE_AND:%.*]] = IE.And([[EXPAND_INPUT]], [[EXPAND_INPUT]]) {auto_broadcast = "NUMPY"}
// CHECK:       [[OUT:%.*]] = IE.Slice [[ELTWISE_AND]] [0, 0, 0, 0] [1, 3, 30, 25]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvolutionChannels
func @ExpandGroupConvolutionChannels(%arg0: tensor<1x72x56x56xf16, {order = #NHWC}>) -> tensor<1x72x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<72x1x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.0> : tensor<72x1x3x3xf16>, [#const.Reorder<#NHWC>]>
    %1 = const.Declare tensor<1x72x1x1xf16> = #const.Content<dense<1.0> : tensor<1x72x1x1xf16>>

    %2 = IE.GroupConvolution(%arg0, %0, %1) {
        dilations = [1, 1], groups = 72, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x72x56x56xf16, {order = #NHWC}>, tensor<72x1x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x72x28x28xf16, {order = #NHWC}>
}

// CHECK:       [[EXTENDED_GROUP:%.*]] = const.Declare tensor<1x80x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]>
// CHECK:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 0, 0, 0]>]>

// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.GroupConvolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_GROUP]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]

// CHECK        return [[REDUNDRANT_SUBTENSOR]]

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
    %filter = const.Declare tensor<3x1x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.0> : tensor<3x1x3x3xf16>, [#const.Reorder<#NHWC>]>
    %1 = IE.GroupConvolution(%input, %filter) {
        dilations = [1, 1], groups = 3, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<3x1x3x3xf16, {order = #NHWC}> -> tensor<1x3x15x15xf16, {order = #NHWC}>
    return %1 : tensor<1x3x15x15xf16, {order = #NHWC}>
}

// CHECK-LABEL: func @ExpandQuantGroupConvolutionChannelsConst
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30xf16, {order = #NHWC}>

// CHECK:       [[EXPAND_FILTER:%.*]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>]>
// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.GroupConvolution([[EXPAND_OUT]], [[EXPAND_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
// CHECK-SAME:      -> tensor<1x16x15x15xf16, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 3, 15, 15]

// CHECK:       return [[SLICE_OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ExpandConvolutionChannelsConst(%input: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]>
    %1 = IE.Convolution(%input, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>
}

// CHECK-LABEL: func @ExpandConvolutionChannelsConst
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30xf16, {order = #NHWC}>

// CHECK:       [[EXPAND_FILTER:%.*]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>
// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.Convolution([[EXPAND_OUT]], [[EXPAND_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
// CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 5, 28, 28]

// CHECK:       return [[SLICE_OUT]] : tensor<1x5x28x28xf16, {order = #NHWC}>
