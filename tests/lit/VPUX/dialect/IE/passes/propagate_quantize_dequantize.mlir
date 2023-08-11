//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-quantize-dequantize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX
!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantReshape
func.func @PropagateDequantReshape(%arg0: tensor<1x256x!qElemType>) -> tensor<1x256x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x!qElemType> -> tensor<1x256xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 256, 1, 1], dim_mapping = [[0], [1, 2, 3]]} : tensor<1x256xf16> -> tensor<1x256x1x1xf16>
  %3 = IE.AffineReshape(%1) {shape_value = [1, 256, 1, 1], dim_mapping = [[0], [1, 2, 3]]} : tensor<1x256xf16> -> tensor<1x256x1x1xf16>
  %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>

  return %4 : tensor<1x256x1x1xf16>

  //CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANT0:%.*]] = IE.Dequantize([[RESHAPE0]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  //CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANT1:%.*]] = IE.Dequantize([[RESHAPE1]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT0]], [[DEQUANT1]])
  //CHECK: return [[ADD]] : tensor<1x256x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantReshape
func.func @PropagateQuantReshape(%arg0: tensor<1x9x1x1xf32>) -> (tensor<1x9x!qElemType>, tensor<1x9x!qElemType>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1x1xf32> -> tensor<1x9x1x1xf16>
  %1 = IE.AffineReshape(%0) {shape_value = [1, 9], dim_mapping = [[0], [1], [1], [1]]} : tensor<1x9x1x1xf16> -> tensor<1x9xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9xf16> -> tensor<1x9x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9xf16> -> tensor<1x9x!qElemType>

  return %2, %3 : tensor<1x9x!qElemType>, tensor<1x9x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 9]} : tensor<1x9x1x1x!qElemType> -> tensor<1x9x!qElemType>
  //CHECK: return [[VAL1]], [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDequantTranspose
func.func @PropagateDequantTranspose(%arg0: tensor<1x256x2x2x!qElemType>) -> tensor<1x2x2x256xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x2x2x!qElemType> -> tensor<1x256x2x2xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x2x256xf16>, tensor<1x2x2x256xf16> -> tensor<1x2x2x256xf16>

  return %3 : tensor<1x2x2x256xf16>

  //CHECK: [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x2x2x256x!qElemType> -> tensor<1x2x2x256xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateQuantTranspose
func.func @PropagateQuantTranspose(%arg0: tensor<1x256x2x2xf32>) -> tensor<1x2x2x256x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x256x2x2xf32> -> tensor<1x256x2x2xf16>
  %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x2x2x256xf16> -> tensor<1x2x2x256x!qElemType>

  return %2 : tensor<1x2x2x256x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x256x2x2xf16> -> tensor<1x256x2x2x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantExpandDilated
func.func @PropagateDequantExpandDilated(%arg0: tensor<1x9x3x3x!qElemType>) -> tensor<1x9x5x5xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x9x5x5xf16>, tensor<1x9x5x5xf16> -> tensor<1x9x5x5xf16>

  return %3 : tensor<1x9x5x5xf16>

  //CHECK: [[VAL0:%.*]] = IE.ExpandDilated(%arg0) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x9x5x5x!qElemType> -> tensor<1x9x5x5xf16>
  //CHECK: [[VAL2:%.*]] = IE.Add
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantConvertExpandDilated
func.func @PropagateDequantConvertExpandDilated(%arg0: tensor<1x9x3x3x!qElemType>) -> tensor<1x9x5x5xf32> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x9x5x5xf16> -> tensor<1x9x5x5xf32>

  return %3 : tensor<1x9x5x5xf32>

  //CHECK: [[VAL0:%.*]] = IE.ExpandDilated(%arg0) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x9x5x5x!qElemType> -> tensor<1x9x5x5xf16>
  //CHECK: [[VAL2:%.*]] = IE.Convert
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantExpandDilated
func.func @PropagateQuantExpandDilated(%arg0: tensor<1x9x3x3xf32>) -> tensor<1x9x5x5x!qElemType> {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x3x3xf32> -> tensor<1x9x3x3xf16>
  %1 = IE.ExpandDilated(%0) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x5x5xf16> -> tensor<1x9x5x5x!qElemType>

  return %2 : tensor<1x9x5x5x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert
  //CHECK: [[VAL0:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x3x3xf16> -> tensor<1x9x3x3x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.ExpandDilated([[VAL0]]) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType0 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagateDequantConstPerAxisReshape
func.func @PropagateDequantConstPerAxisReshape() -> tensor<3x1x1x1xf16> {
  %0 = const.Declare tensor<1x3x1x1x!qElemType1> = dense<1.0> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x1x1x!qElemType1> -> tensor<1x3x1x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [3, 1, 1, 1], dim_mapping = [[0], [0], [1], [2, 3]]} : tensor<1x3x1x1xf16> -> tensor<3x1x1x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>

  return %3 : tensor<3x1x1x1xf16>

  //CHECK: [[CONST:%.*]] =  const.Declare tensor<3x1x1x1x!qElemType0> = dense<1.000000e+00> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reshape<[3, 1, 1, 1]>]
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<3x1x1x1x!qElemType0> -> tensor<3x1x1x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<3x1x1x1xf16>
}

// -----

!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType0 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagationDequantPerAxisReshapeOneToZeroAxis
func.func @PropagationDequantPerAxisReshapeOneToZeroAxis(%arg0: tensor<1x3x2x1x!qElemType0>) -> tensor<3x2x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x2x1x!qElemType0> -> tensor<1x3x2x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [3, 2, 1, 1], dim_mapping = [[0], [0], [1], [2, 3]]} : tensor<1x3x2x1xf16> -> tensor<3x2x1x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x2x1x1xf16>, tensor<3x2x1x1xf16> -> tensor<3x2x1x1xf16>

  return %3 : tensor<3x2x1x1xf16>

  //CHECK: [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [3, 2, 1, 1]} : tensor<1x3x2x1x!qElemType0> -> tensor<3x2x1x1x!qElemType1>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[RESHAPE]]) {dstElemType = f16} : tensor<3x2x1x1x!qElemType1> -> tensor<3x2x1x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<3x2x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisReshape
func.func @NoPropagationDequantPerAxisReshape(%arg0: tensor<1x3x1x2x!qElemType>) -> tensor<1x2x1x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %2 = IE.Reshape(%1) {shape_value = [1, 2, 1, 3]} : tensor<1x3x1x2xf16> -> tensor<1x2x1x3xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x3xf16>, tensor<1x2x1x3xf16> -> tensor<1x2x1x3xf16>

  return %3 : tensor<1x2x1x3xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  //CHECK: [[RESHAPE:%.*]] = IE.Reshape([[DEQUANT]]) {shape_value = [1, 2, 1, 3]} : tensor<1x3x1x2xf16> -> tensor<1x2x1x3xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<1x2x1x3xf16>
}

// -----

!qElemType0 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagationDequantPerAxisReshapeZeroToOneAxis
func.func @PropagationDequantPerAxisReshapeZeroToOneAxis(%arg0: tensor<3x2x1x1x!qElemType0>) -> tensor<1x3x2x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<3x2x1x1x!qElemType0> -> tensor<3x2x1x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 3, 2, 1], dim_mapping = [[0, 1], [2], [3], [3]]} : tensor<3x2x1x1xf16> -> tensor<1x3x2x1xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x2x1xf16>, tensor<1x3x2x1xf16> -> tensor<1x3x2x1xf16>

  return %3 : tensor<1x3x2x1xf16>

  //CHECK: [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 3, 2, 1]} : tensor<3x2x1x1x!qElemType0> -> tensor<1x3x2x1x!qElemType1>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x2x1x!qElemType1> -> tensor<1x3x2x1xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<1x3x2x1xf16>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagateDequantConstPerAxisExpandDilated
func.func @PropagateDequantConstPerAxisExpandDilated() -> tensor<1x3x5x5xf16> {
  %0 = const.Declare tensor<1x3x3x3x!qElemType> = dense<1.0> : tensor<1x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x3x3x!qElemType> -> tensor<1x3x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x5xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x5x5xf16>, tensor<1x3x5x5xf16> -> tensor<1x3x5x5xf16>

  return %3 : tensor<1x3x5x5xf16>

  //CHECK: [[CONST:%.*]] =  const.Declare tensor<1x3x5x5x!qElemType> = dense<1.000000e+00> : tensor<1x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.ExpandDilated<[2, 2]>]
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<1x3x5x5x!qElemType> -> tensor<1x3x5x5xf16>
  //CHECK: [[ADD:%.*]] = IE.Add
  //CHECK: return [[ADD]] : tensor<1x3x5x5xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

// CHECK-LABEL: @PropagateQuantAffineReshapeTranspose
func.func @PropagateQuantAffineReshapeTranspose(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4x!qElemType> {
  %0 = IE.AffineReshape(%arg0) {shape_value = [1, 3, 4, 1, 4, 1], dim_mapping = [[0], [1], [2, 3], [4, 5]]} : tensor<1x3x4x4xf16> -> tensor<1x3x4x1x4x1xf16>
  %1 = IE.Transpose(%0) {order_value = #map} : tensor<1x3x4x1x4x1xf16> -> tensor<1x1x1x3x4x4xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 3, 4, 4], dim_mapping = [[0], [0], [0], [1], [2], [3]]} : tensor<1x1x1x3x4x4xf16> -> tensor<1x3x4x4xf16>
  %3 = IE.Quantize(%2) {dstElemType = !qElemType} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4x!qElemType>

  return %3 : tensor<1x3x4x4x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4x!qElemType>
  //CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape([[QUANTIZE]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [4, 5]], shape_value = [1, 3, 4, 1, 4, 1]} : tensor<1x3x4x4x!qElemType> -> tensor<1x3x4x1x4x1x!qElemType>
  //CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map}
  //CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1], [2], [3]], shape_value = [1, 3, 4, 4]} : tensor<1x1x1x3x4x4x!qElemType> -> tensor<1x3x4x4x!qElemType>
  //CHECK: return [[RESHAPE1]] : tensor<1x3x4x4x!qElemType>
}

// ######## CONCAT TEST ########

// -----

!qElemType0 = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcat
func.func @PerTensorConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType0>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 1} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxisConcat
func.func @PerAxisQuantOtherAxisConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType0>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 2} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxisConcat
func.func @PerAxisQuantSameAxisConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    // expected-error@+1 {{Misaligned element types}}
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 1} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----
!qElemType0 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @NoPropagationPerTensorQuantConcat
func.func @NoPropagationPerTensorQuantConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 1} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]]) {per_axis = {axis = 1 : i64}} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @QuantizePropagationAndNoDequantizePropagationPerTensorQuantConcat
func.func @QuantizePropagationAndNoDequantizePropagationPerTensorQuantConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType0> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 1} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType0} : tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType0>
    return %3 : tensor<1x4x3x4x!qElemType0>

    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[QUANTIZE1:%.*]] = IE.Quantize([[DEQUANTIZE1]]) {dstElemType = !qElemType0} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType0>
    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, [[QUANTIZE1]]) {per_axis = {axis = 1 : i64}} : tensor<1x2x3x4x!qElemType0>, tensor<1x2x3x4x!qElemType0> -> tensor<1x4x3x4x!qElemType0>

    //CHECK: return [[CONCAT]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantOtherAxisConcat
func.func @NoPropagationPerAxisQuantOtherAxisConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x2x6x4x!qElemType0> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 2} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType0} : tensor<1x2x6x4xf16> -> tensor<1x2x6x4x!qElemType0>
    return %3 : tensor<1x2x6x4x!qElemType0>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONCAT]])
    //CHECK: return [[QUANTIZE]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantSameAxisConcat
func.func @NoPropagationPerAxisQuantSameAxisConcat(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType2> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %2 = IE.Concat(%0, %1) { per_axis = {axis = 1} } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType2} : tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType2>
    return %3 : tensor<1x4x3x4x!qElemType2>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONCAT]])
    //CHECK: return [[QUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcatOffsets
func.func @PerTensorConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxisConcatOffsets
func.func @PerAxisQuantOtherAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxisConcatOffsets
func.func @PerAxisQuantSameAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[CONCAT:%.*]] = IE.Concat(%arg0, %arg1)
    //CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[CONCAT]])
    //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE]], [[DEQUANTIZE]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @NoPropagationPerTensorConcatOffsets
func.func @NoPropagationPerTensorConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantOtherAxisConcatOffsets
func.func @NoPropagationPerAxisQuantOtherAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x2x6x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x4xf16>
    return %3 : tensor<1x2x6x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisQuantSameAxisConcatOffsets
func.func @NoPropagationPerAxisQuantSameAxisConcatOffsets(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType0> -> tensor<1x2x3x4xf16>
    %1 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %3 : tensor<1x4x3x4xf16>

    //CHECK: [[DEQUANTIZE0:%.*]] = IE.Dequantize(%arg0)
    //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize(%arg1)
    //CHECK: [[CONCAT:%.*]] = IE.Concat([[DEQUANTIZE0]], [[DEQUANTIZE1]])
    //CHECK: [[ADD:%.*]] = IE.Add([[CONCAT]], [[CONCAT]])
    //CHECK: return [[ADD]]
}

// ######## CONCAT TEST ########

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantClamp
func.func @PropagateDequantClamp(%arg0: tensor<1x256x1x1x!qElemType>) -> tensor<1x256x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  %2 = IE.Clamp(%1) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>
  %3 = IE.Clamp(%1) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>
  %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>

  return %4 : tensor<1x256x1x1xf16>

  //CHECK: [[CLAMP1:%.*]] = IE.Clamp(%arg0) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANTIZE1:%.*]] = IE.Dequantize([[CLAMP1]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>

  //CHECK: [[CLAMP2:%.*]] = IE.Clamp(%arg0) {max = 5.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANTIZE2:%.*]] = IE.Dequantize([[CLAMP2]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>

  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANTIZE1]], [[DEQUANTIZE2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1xf16>

  //CHECK: return [[ADD]] : tensor<1x256x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantClamp
func.func @PropagateQuantClamp(%arg0: tensor<1x9x1x1xf32>) -> (tensor<1x9x1x1x!qElemType>, tensor<1x9x1x1x!qElemType>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1x1xf32> -> tensor<1x9x1x1xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>

  return %2, %3 : tensor<1x9x1x1x!qElemType>, tensor<1x9x1x1x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x9x1x1xf32> -> tensor<1x9x1x1xf16>
  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[QUANTIZE]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x9x1x1x!qElemType> -> tensor<1x9x1x1x!qElemType>

  //CHECK: return [[CLAMP]], [[CLAMP]] : tensor<1x9x1x1x!qElemType>, tensor<1x9x1x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisClamp
func.func @NoPropagationDequantPerAxisClamp(%arg0: tensor<1x3x1x2x!qElemType>) -> tensor<1x3x1x2xf16> {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>

  return %2 : tensor<1x3x1x2xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0)
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[DEQUANT]])
  //CHECK: [[ADD:%.*]] = IE.Add([[CLAMP]], [[CLAMP]])
  //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisClamp
func.func @NoPropagationDequantPerAxisClamp(%arg0: tensor<1x3x1x2xf32>) -> (tensor<1x3x1x2x!qElemType>, tensor<1x3x1x2x!qElemType>) {
  %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1x2xf32> -> tensor<1x3x1x2xf16>
  %1 = IE.Clamp(%0) {max = 6.000000e+00, min = 0.000000e+00} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2x!qElemType>

  return %2, %3 : tensor<1x3x1x2x!qElemType>, tensor<1x3x1x2x!qElemType>

  //CHECK: [[CONVERT:%.*]] = IE.Convert(%arg0)
  //CHECK: [[CLAMP:%.*]] = IE.Clamp([[CONVERT]])
  //CHECK: [[QUANTIZE0:%.*]] = IE.Quantize([[CLAMP]])
  //CHECK: [[QUANTIZE1:%.*]] = IE.Quantize([[CLAMP]])
  //CHECK: return [[QUANTIZE0]], [[QUANTIZE1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantMaxPool
func.func @PropagateDequantMaxPool(%arg0: tensor<1x512x19x19x!qElemType>) -> tensor<1x512x19x19xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19xf16>
  %2 = IE.MaxPool(%1) {
        kernel_size = [13, 13],
        pads_begin = [6, 6],
        pads_end = [6, 6],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x512x19x19xf16>, tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>

  return %3 : tensor<1x512x19x19xf16>

  //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool(%arg0) {kernel_size = [13, 13], pads_begin = [6, 6], pads_end = [6, 6], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[MAXPOOL]]) {dstElemType = f16} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19xf16>
  //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x19x19xf16>, tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
  //CHECK: return [[ADD]] : tensor<1x512x19x19xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

 // CHECK-LABEL: @PropagateQuantMaxPool
func.func @PropagateQuantMaxPool(%arg0: tensor<1x512x19x19xf16>) -> tensor<1x512x19x19x!qElemType> {
  %1 = IE.MaxPool(%arg0) {
        kernel_size = [13, 13],
        pads_begin = [6, 6],
        pads_end = [6, 6],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19x!qElemType>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}  : tensor<1x512x19x19x!qElemType>, tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>

  return %3 : tensor<1x512x19x19x!qElemType>

  //CHECK: [[QUANTIZE:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x512x19x19xf16> -> tensor<1x512x19x19x!qElemType>
  //CHECK: [[MAXPOOL:%.*]] = IE.MaxPool([[QUANTIZE]]) {kernel_size = [13, 13], pads_begin = [6, 6], pads_end = [6, 6], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>
  //CHECK: [[ADD:%.*]] = IE.Add([[MAXPOOL]], [[MAXPOOL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x19x19x!qElemType>, tensor<1x512x19x19x!qElemType> -> tensor<1x512x19x19x!qElemType>
  //CHECK: return [[ADD]] : tensor<1x512x19x19x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateDequantSplit
func.func @PropagateDequantSplit(%arg0: tensor<1x3x30x30x!qElemType>) -> (tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>) {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x30x30x!qElemType> -> tensor<1x3x30x30xf16>
  %1:3 = IE.Split(%0) {axis_value = 1, num_splits = 3} : tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
  %2 = IE.Add(%1#0, %1#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
  %3 = IE.Add(%1#1, %1#1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
  %4 = IE.Add(%1#2, %1#2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>

 return %2, %3, %4 : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

 //CHECK: [[SPLIT:%.*]]:3 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 3 : i64} : tensor<1x3x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>
 //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[SPLIT]]#2) {dstElemType = f16} : tensor<1x1x30x30x!qElemType> -> tensor<1x1x30x30xf16>
 //CHECK: [[DEQUANT1:%.*]] = IE.Dequantize([[SPLIT]]#1) {dstElemType = f16} : tensor<1x1x30x30x!qElemType> -> tensor<1x1x30x30xf16>
 //CHECK: [[DEQUANT2:%.*]] = IE.Dequantize([[SPLIT]]#0) {dstElemType = f16} : tensor<1x1x30x30x!qElemType> -> tensor<1x1x30x30xf16>
 //CHECK: [[ADD:%.*]] = IE.Add([[DEQUANT2]], [[DEQUANT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
 //CHECK: [[ADD1:%.*]] = IE.Add([[DEQUANT1]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
 //CHECK: [[ADD2:%.*]] = IE.Add([[DEQUANT]], [[DEQUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x1x30x30xf16>
 //CHECK: return [[ADD]], [[ADD1]], [[ADD2]] : tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>

}

// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisSplit
func.func @NoPropagationDequantPerAxisSplit(%arg0: tensor<1x3x1x2x!qElemType>) -> (tensor<1x3x1x2xf16>) {
  %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %1:1 = IE.Split(%0) {axis_value = 1, num_splits = 1} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  %2 = IE.Add(%1#0, %1#0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>

  return %2 : tensor<1x3x1x2xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  //CHECK: [[SPLIT:%.*]] = IE.Split([[DEQUANT]]) {axis_value = 1 : i64, num_splits = 1 : i64} : tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  //CHECK: [[ADD:%.*]] = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x1x2xf16>, tensor<1x3x1x2xf16> -> tensor<1x3x1x2xf16>
  //CHECK: return [[ADD]] : tensor<1x3x1x2xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantSplit
func.func @PropagateQuantSplit(%arg0: tensor<1x3x30x30xf16>) -> (tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>) {
  %0:3 = IE.Split(%arg0) {axis_value = 1, num_splits = 3} : tensor<1x3x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
  %1 = IE.Quantize(%0#0) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
  %2 = IE.Quantize(%0#1) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
  %3 = IE.Quantize(%0#2) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>

 return %1, %2, %3 : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>

 //CHECK: [[QUANT:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30x!qElemType>
 //CHECK: [[SPLIT:%.*]]:3 = IE.Split([[QUANT]]) {axis_value = 1 : i64, num_splits = 3 : i64} : tensor<1x3x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>
 //CHECK: return [[SPLIT]]#0, [[SPLIT]]#1, [[SPLIT]]#2 : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @FuseQuantParamsIntoSplit
func.func @FuseQuantParamsIntoSplit(%arg0: tensor<1x2x3x4xf16>) -> (tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>) {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2:2 = IE.Split(%1) {num_splits = 2, axis_value = 1} : tensor<1x2x3x4xf16> -> tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>

    %3 = IE.Quantize(%2#0) {dstElemType = !qElemType}: tensor<1x1x3x4xf16> -> tensor<1x1x3x4x!qElemType>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>

    %5 = IE.Quantize(%2#1) {dstElemType = !qElemType}: tensor<1x1x3x4xf16> -> tensor<1x1x3x4x!qElemType>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>

    return %4, %6 : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    //CHECK: [[VAL1:%.*]]:2 = IE.Split([[VAL0]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x3x4x!qElemType> -> tensor<1x1x3x4x!qElemType>, tensor<1x1x3x4x!qElemType>
    //CHECK: [[VAL2:%.*]] = IE.Dequantize([[VAL1]]#0) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>
    //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL1]]#1) {dstElemType = f16} : tensor<1x1x3x4x!qElemType> -> tensor<1x1x3x4xf16>
    //CHECK: return [[VAL2]], [[VAL3]] : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>
!qElemType1 = !quant.uniform<i8:f16, 0.0016544117647058823>

// CHECK-LABEL: @NoPropagateQuantSplitDifferentTypes
func.func @NoPropagateQuantSplitDifferentTypes(%arg0: tensor<1x2x30x30xf16>) -> (tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType1>) {
  %0:2 = IE.Split(%arg0) {axis_value = 1, num_splits = 2} : tensor<1x2x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
  %1 = IE.Quantize(%0#0) {dstElemType = !qElemType} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType>
  %2 = IE.Quantize(%0#1) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>

 return %1, %2 : tensor<1x1x30x30x!qElemType>, tensor<1x1x30x30x!qElemType1>

 //CHECK: [[SPLIT:%.*]]:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x30x30xf16> -> tensor<1x1x30x30xf16>, tensor<1x1x30x30xf16>
 //CHECK: [[QUANT0:%.*]] = IE.Quantize([[SPLIT]]#0) {dstElemType = !qElemType0} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType0>
 //CHECK: [[QUANT1:%.*]] = IE.Quantize([[SPLIT]]#1) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
 //CHECK: return [[QUANT0]], [[QUANT1]] : tensor<1x1x30x30x!qElemType0>, tensor<1x1x30x30x!qElemType1>

}
