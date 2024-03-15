//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-quantized-ops="se-transposed-conv-enabled=true" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FuseQuantParamsIntoTransposedConv
func.func @FuseQuantParamsIntoTransposedConv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
  %weights = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType> -> tensor<3x3x4x4xf16>
  %4 = IE.TransposedConvolution(%2, %3) {
      dilations = [1, 1],
      operandSegmentSizes = array<i32: 1, 1, 0, 0>,
      output_padding = [0, 0],
      pads_begin = [1, 1],
      pads_end = [1, 1],
      strides = [2, 2]
    } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
  %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

  return %6 : tensor<1x3x32x32xf16>

  //CHECK: [[CST:%.*]] = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.000000e+00> : tensor<3x3x4x4xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
  //CHECK: [[VAL2:%.*]] = IE.TransposedConvolution([[VAL1]], [[CST]]) {
  //CHECK-SAME:   dilations = [1, 1],
  //CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
  //CHECK-SAME:   output_padding = [0, 0],
  //CHECK-SAME:   pads_begin = [1, 1],
  //CHECK-SAME:   pads_end = [1, 1],
  //CHECK-SAME:   strides = [2, 2]
  //CHECK-SAME:   } : tensor<1x3x16x16x!qElemType1>, tensor<3x3x4x4x!qElemType>
  //CHECK-SAME:    -> tensor<1x3x32x32x!qElemType2>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>
  //CHECK: return [[VAL3]]
}
