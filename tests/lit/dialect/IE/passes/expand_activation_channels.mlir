// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB compilation-mode=ReferenceHW" --expand-activation-channels --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ExpandMaxPoolChannels
func @ExpandMaxPoolChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x15x13xf16> {
  %0 = IE.MaxPool(%arg0) {kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = "FLOOR", strides = [2, 2]} : tensor<1x3x30x30xf16> -> tensor<1x3x15x13xf16>
  // CHECK:       %[[PAD:.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x30xf16> -> tensor<1x16x30x30xf16>
  // CHECK:       %[[POOL:.*]] = IE.MaxPool(%[[PAD]]) {kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = "FLOOR", strides = [2, 2]} : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>
  // CHECK:       %[[OUT:.*]] = IE.Slice %[[POOL]] [0, 0, 0, 0] [1, 3, 15, 13] : tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

  return %0 : tensor<1x3x15x13xf16>
  // CHECK        return %[[OUT]]
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

func @ExpandQuantMaxPoolChannels(
        %input: tensor<1x3x30x30x!qElemType0>)
            -> tensor<1x3x15x13x!qElemType1> {
    %1 = IE.MaxPool(%input) { kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = "FLOOR", strides = [2, 2] }
        : tensor<1x3x30x30x!qElemType0> -> tensor<1x3x15x13x!qElemType1>
    return %1 : tensor<1x3x15x13x!qElemType1>
}
// CHECK-LABEL: func @ExpandQuantMaxPoolChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>

// CHECK: [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
// CHECK-SAME: tensor<1x3x30x30x!quant.uniform<u8:f16, 0.96372549019607844>> -> tensor<1x16x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>

// CHECK: [[MAXPOOL_OUT:%.+]] = IE.MaxPool([[EXPAND_OUT]]) {kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = "FLOOR", strides = [2, 2]} :
// CHECK-SAME: tensor<1x16x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>
// CHECK-SAME: -> tensor<1x16x15x13x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>

// CHECK: [[SLICE_OUT:%.+]] = IE.Slice [[MAXPOOL_OUT]] [0, 0, 0, 0] [1, 3, 15, 13] :
// CHECK-SAME: tensor<1x16x15x13x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>
// CHECK-SAME: to tensor<1x3x15x13x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>>

// CHECK: return [[SLICE_OUT]] : tensor<1x3x15x13x!quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>>

// -----

// CHECK-LABEL: @ExpandConvolutionChannels
func @ExpandConvolutionChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x5x28x28xf16> {
  %0 = const.Declare tensor<5x3x3x3xf16> = #const.Content<dense<1.0> : tensor<5x3x3x3xf16>>

  // CHECK:       %[[EXTENDED_FILTER:.*]] = const.Declare tensor<16x16x3x3xf16> =
  // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<5x3x3x3xf16>, [#const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>
  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<5x3x3x3xf16> -> tensor<1x5x28x28xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.Convolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = IE.Slice %[[EXTENDED_CONV]]

  return %1 : tensor<1x5x28x28xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>

func @ExpandQuantConvolutionChannels(
        %input: tensor<1x3x30x30x!qElemType0>,
        %filter: tensor<5x3x3x3x!qElemType1>)
            -> tensor<1x5x28x28x!qElemType2> {
    %1 = IE.Convolution(%input, %filter) { dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1] }
        : tensor<1x3x30x30x!qElemType0>, tensor<5x3x3x3x!qElemType1> -> tensor<1x5x28x28x!qElemType2>
    return %1 : tensor<1x5x28x28x!qElemType2>
}

// CHECK-LABEL: func @ExpandQuantConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<5x3x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>>

// CHECK: [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x30x!quant.uniform<u8:f16, 0.96372549019607844>> -> tensor<1x16x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>
// CHECK: [[PAD_OUT:%.+]] = IE.Pad([[FILTER]]) {mode = "CONSTANT", pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [11, 13, 0, 0]} : tensor<5x3x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>> -> tensor<16x16x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>

// CHECK: [[CONV_OUT:%.+]] = IE.Convolution([[EXPAND_OUT]], [[PAD_OUT]])
// CHECK-SAME: {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
// CHECK-SAME: tensor<1x16x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>,
// CHECK-SAME: tensor<16x16x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>
// CHECK-SAME: -> tensor<1x16x28x28x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>

// CHECK: [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 5, 28, 28] :
// CHECK-SAME: tensor<1x16x28x28x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>
// CHECK-SAME: to tensor<1x5x28x28x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>>

// CHECK: return [[SLICE_OUT]] : tensor<1x5x28x28x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>>

// -----

// CHECK-LABEL: @ExpandBiasesConvolutionChannels
func @ExpandBiasesConvolutionChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x5x28x28xf16> {
  %0 = const.Declare tensor<5x3x3x3xf16> = #const.Content<dense<1.0> : tensor<5x3x3x3xf16>>
  %1 = const.Declare tensor<1x5x1x1xf16> = #const.Content<dense<1.0> : tensor<1x5x1x1xf16>>

  // CHECK-DAG:   %[[EXTENDED_FILTER:.*]] = const.Declare tensor<16x16x3x3xf16> = #const.Content<dense<1.000000e+00> : tensor<5x3x3x3xf16>, [#const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>
  // CHECK-DAG:   %[[EXTENDED_BIAS:.*]] = const.Declare tensor<1x16x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x5x1x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]>

  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %2 = IE.Convolution(%arg0, %0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<5x3x3x3xf16>, tensor<1x5x1x1xf16> -> tensor<1x5x28x28xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.Convolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]], %[[EXTENDED_BIAS]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = IE.Slice %[[EXTENDED_CONV]]

  return %2 : tensor<1x5x28x28xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}

// -----

// CHECK-LABEL: @ExpandEltwiseAddChannels
func @ExpandEltwiseAddChannels(%arg0: tensor<1x3x30x25xf16>, %arg1: tensor<1x3x30x25xf16>) -> tensor<1x3x30x25xf16> {
  %0 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x3x30x25xf16>, tensor<1x3x30x25xf16> -> tensor<1x3x30x25xf16>
  // CHECK:       %[[EXPAND_LEFT_INPUT:.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x25xf16> -> tensor<1x16x30x25xf16>
  // CHECK:       %[[EXPAND_RIGHT_INPUT:.*]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x25xf16> -> tensor<1x16x30x25xf16>
  // CHECK:       %[[ELTWISE_ADD:.*]] = IE.Add(%[[EXPAND_LEFT_INPUT]], %[[EXPAND_RIGHT_INPUT]]) {auto_broadcast = "NUMPY"} : tensor<1x16x30x25xf16>, tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16>
  // CHECK:       %[[OUT:.*]] = IE.Slice %[[ELTWISE_ADD]] [0, 0, 0, 0] [1, 3, 30, 25] : tensor<1x16x30x25xf16> to tensor<1x3x30x25xf16>

  return %0 : tensor<1x3x30x25xf16>
  // CHECK        return %[[OUT]]
}

// -----

// CHECK-LABEL: @ExpandGroupConvolutionChannels
func @ExpandGroupConvolutionChannels(%arg0: tensor<1x72x56x56xf16>) -> tensor<1x72x28x28xf16> {
  %0 = const.Declare tensor<72x1x3x3xf16> = #const.Content<dense<1.0> : tensor<72x1x3x3xf16>>
  %1 = const.Declare tensor<1x72x1x1xf16> = #const.Content<dense<1.0> : tensor<1x72x1x1xf16>>

  // CHECK:       %[[EXTENDED_GROUP:.*]] = const.Declare tensor<1x80x1x1xf16> =
  // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<1x72x1x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]>
  // CHECK:       %[[EXTENDED_FILTER:.*]] = const.Declare tensor<80x1x3x3xf16> =
  // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<72x1x3x3xf16>, [#const.PadWithZero<[0, 0, 0, 0], [8, 0, 0, 0]>]>

  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %2 = IE.GroupConvolution(%arg0, %0, %1) {dilations = [1, 1], groups = 72, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]} : tensor<1x72x56x56xf16>, tensor<72x1x3x3xf16>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.GroupConvolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]], %[[EXTENDED_GROUP]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = IE.Slice %[[EXTENDED_CONV]]

  return %2 : tensor<1x72x28x28xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127}>

func @ExpandQuantGroupConvolutionChannels(
        %input: tensor<1x3x30x30x!qElemType0>,
        %filter: tensor<3x1x3x3x!qElemType1>)
            -> tensor<1x3x15x15x!qElemType2> {
    %1 = IE.GroupConvolution(%input, %filter) { dilations = [1, 1], groups = 3, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2] }
        : tensor<1x3x30x30x!qElemType0>, tensor<3x1x3x3x!qElemType1> -> tensor<1x3x15x15x!qElemType2>
    return %1 : tensor<1x3x15x15x!qElemType2>
}

// CHECK-LABEL: func @ExpandQuantGroupConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<3x1x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>>

// CHECK: [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x30x!quant.uniform<u8:f16, 0.96372549019607844>> -> tensor<1x16x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>
// CHECK: [[PAD_OUT:%.+]] = IE.Pad([[FILTER]]) {mode = "CONSTANT", pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [13, 0, 0, 0]} : tensor<3x1x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>> -> tensor<16x1x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>

// CHECK: [[CONV_OUT:%.+]] = IE.GroupConvolution([[EXPAND_OUT]], [[PAD_OUT]])
// CHECK-SAME: {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]} :
// CHECK-SAME: tensor<1x16x30x30x!quant.uniform<u8:f16, 0.96372549019607844>>,
// CHECK-SAME: tensor<16x1x3x3x!quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>
// CHECK-SAME: -> tensor<1x16x15x15x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>

// CHECK: [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 3, 15, 15] :
// CHECK-SAME: tensor<1x16x15x15x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>>
// CHECK-SAME: to tensor<1x3x15x15x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127}>>

// CHECK: return [[SLICE_OUT]] : tensor<1x3x15x15x!quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127}>>
