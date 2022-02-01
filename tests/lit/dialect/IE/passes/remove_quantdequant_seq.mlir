// RUN: vpux-opt --split-input-file --remove-quantdequant-seq %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = type !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = type !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @RemoveQuantDequantSequence
func @RemoveQuantDequantSequence(%arg0: tensor<1x3x16x16xf16>) -> (tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>)  {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x3x3x!qElemType0> = #const.Content<dense<1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType0> -> tensor<3x3x3x3xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

  return %6, %4 : tensor<1x3x14x14xf16>, tensor<1x3x14x14xf16>

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType0} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType0>
  //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType1> =
  //CHECK-SAME:                 #const.Content<dense<1.000000e+00> : tensor<3x3x3x3xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]>
  //CHECK: [[VAL2:%.*]] = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType1> -> tensor<3x3x3x3xf16>

  //CHECK: [[VAL3:%.*]] = IE.Convolution(%arg0, [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
  //CHECK: return [[VAL3]], [[VAL3]]
}
