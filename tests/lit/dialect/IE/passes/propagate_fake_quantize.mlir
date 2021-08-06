// RUN: vpux-opt --split-input-file --propagate-quantize-dequantize %s | FileCheck %s

// CHECK-LABEL: @FuseQuantParamsIntoConv
func @FuseQuantParamsIntoConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = "quant.qcast"(%arg0) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  %2 = "quant.dcast"(%1) : (tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>) -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> = #const.Content<dense<1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>>]>
  %3 = "quant.dcast"(%weights) : (tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>>) -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = "quant.qcast"(%4) : (tensor<1x3x14x14xf16>) -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
  %6 = "quant.dcast"(%5) : (tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>) -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>
  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> =
  //CHECK-SAME:                 #const.Content<dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>>]

  //CHECK: [[VAL1:%.*]] = "quant.qcast"(%arg0) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>, tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
  //CHECK: [[VAL3:%.*]] = "quant.dcast"([[VAL2]]) : (tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>) -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]]
}

// CHECK-LABEL: @FuseQuantParamsIntoPool
func @FuseQuantParamsIntoPool(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = "quant.qcast"(%arg0) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  %2 = "quant.dcast"(%1) : (tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>) -> tensor<1x3x16x16xf16>
  %3 = IE.MaxPool(%2) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  %4 = "quant.qcast"(%3) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  %5 = "quant.dcast"(%4) : (tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>) -> tensor<1x3x16x16xf16>
  return %5 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = "quant.qcast"(%arg0) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  //CHECK: [[VAL1:%.*]] = IE.MaxPool([[VAL0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>
  //CHECK: [[VAL2:%.*]] = "quant.dcast"([[VAL1]]) : (tensor<1x3x16x16x!quant.uniform<u8:f16, 0.57450980392156858>>) -> tensor<1x3x16x16xf16>
  //CHECK: return [[VAL2]]
}
