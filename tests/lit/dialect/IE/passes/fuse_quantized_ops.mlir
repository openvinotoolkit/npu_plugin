// RUN: vpux-opt --split-input-file --fuse-quantized-ops %s | FileCheck %s

// CHECK-LABEL: @FuseQuantParamsIntoConv
func @FuseQuantParamsIntoConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> = #const.Content<dense<1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>>]>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !quant.uniform<u8:f16, 2.4627450980392158>}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>> -> tensor<1x3x14x14xf16>

  return %6 : tensor<1x3x14x14xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> =
  //CHECK-SAME:                 #const.Content<dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>, tensor<3x3x3x3x!quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]]
}

// CHECK-LABEL: @FuseQuantParamsIntoSlice
func @FuseQuantParamsIntoSlice(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x8xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x16xf16>
    %2 = IE.Slice %1 [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    %3 = IE.Quantize(%2) {dstElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>}: tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x8xf16>

    return %4 : tensor<1x3x16x8xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL1:%.*]] = IE.Slice [[VAL0]] [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195:128>> to tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x3x16x8x!quant.uniform<u8:f16, 1.1534313725490195:128>> -> tensor<1x3x16x8xf16>
    //CHECK: return [[VAL2]] : tensor<1x3x16x8xf16>
}
