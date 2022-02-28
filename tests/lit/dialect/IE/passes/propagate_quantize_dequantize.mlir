// RUN: vpux-opt --split-input-file --propagate-quantize-dequantize %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantReshape
func @PropagateDequantReshape(%arg0: tensor<1x256x!qElemType>) -> (tensor<1x1x1x256xf16>, tensor<1x256x1x1xf16>) {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x!qElemType> -> tensor<1x256xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 1, 1, 256], dim_mapping = [[0, 1, 2], [3]]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>
  %3 = IE.AffineReshape(%1) {shape_value = [1, 256, 1, 1], dim_mapping = [[0], [1, 2, 3]]} : tensor<1x256xf16> -> tensor<1x256x1x1xf16>

  return %2, %3 : tensor<1x1x1x256xf16>, tensor<1x256x1x1xf16>

  //CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256x!qElemType> -> tensor<1x1x1x256x!qElemType>
  //CHECK: [[DEQUANT0:%.*]] = IE.Dequantize([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x1x256x!qElemType> -> tensor<1x1x1x256xf16>
  //CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x256x!qElemType> -> tensor<1x256x1x1x!qElemType>
  //CHECK: [[DEQUANT1:%.*]] = IE.Dequantize([[RESHAPE1]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xf16>
  //CHECK: return [[DEQUANT0]], [[DEQUANT1]] : tensor<1x1x1x256xf16>, tensor<1x256x1x1xf16>
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantReshape
func @PropagateQuantReshape(%arg0: tensor<1x9x1x1xf16>) -> (tensor<1x9x!qElemType>, tensor<1x9x!qElemType>) {
  %1 = IE.AffineReshape(%arg0) {shape_value = [1, 9], dim_mapping = [[0], [1], [1], [1]]} : tensor<1x9x1x1xf16> -> tensor<1x9xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9xf16> -> tensor<1x9x!qElemType>
  %3 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9xf16> -> tensor<1x9x!qElemType>

  return %2, %3 : tensor<1x9x!qElemType>, tensor<1x9x!qElemType>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x9x1x1xf16> -> tensor<1x9x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 9]} : tensor<1x9x1x1x!qElemType> -> tensor<1x9x!qElemType>
  //CHECK: return [[VAL1]], [[VAL1]]
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDequantTranspose
func @PropagateDequantTranspose(%arg0: tensor<1x256x2x2x!qElemType>) -> tensor<1x2x2x256xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x2x2x!qElemType> -> tensor<1x256x2x2xf16>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>

  return %2 : tensor<1x2x2x256xf16>

  //CHECK: [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x2x2x256x!qElemType> -> tensor<1x2x2x256xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.0016649433210784313>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateQuantTranspose
func @PropagateQuantTranspose(%arg0: tensor<1x256x2x2xf16>) -> tensor<1x2x2x256x!qElemType> {
  %1 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x256x2x2xf16> -> tensor<1x2x2x256xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x2x2x256xf16> -> tensor<1x2x2x256x!qElemType>

  return %2 : tensor<1x2x2x256x!qElemType>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x256x2x2xf16> -> tensor<1x256x2x2x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NHWC}
  //CHECK-SAME: : tensor<1x256x2x2x!qElemType> -> tensor<1x2x2x256x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.0016544117647058823>

// CHECK-LABEL: @PropagateDequantExpandDilated
func @PropagateDequantExpandDilated(%arg0: tensor<1x9x3x3x!qElemType>) -> tensor<1x9x5x5xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>

  return %2 : tensor<1x9x5x5xf16>

  //CHECK: [[VAL0:%.*]] = IE.ExpandDilated(%arg0) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x9x5x5x!qElemType> -> tensor<1x9x5x5xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.0024337469362745098>

// CHECK-LABEL: @PropagateQuantExpandDilated
func @PropagateQuantExpandDilated(%arg0: tensor<1x9x3x3xf16>) -> tensor<1x9x5x5x!qElemType> {
  %1 = IE.ExpandDilated(%arg0) {dilations = [2, 2]} : tensor<1x9x3x3xf16> -> tensor<1x9x5x5xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemType} : tensor<1x9x5x5xf16> -> tensor<1x9x5x5x!qElemType>

  return %2 : tensor<1x9x5x5x!qElemType>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x9x3x3xf16> -> tensor<1x9x3x3x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.ExpandDilated([[VAL0]]) {dilations = [2, 2]} : tensor<1x9x3x3x!qElemType> -> tensor<1x9x5x5x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagateDequantConstPerAxisReshape
func @PropagateDequantConstPerAxisReshape() -> tensor<3x1x1x1xf16> {
  %0 = const.Declare tensor<1x3x1x1x!qElemType1> = #const.Content<dense<1.0> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x1x1x!qElemType1> -> tensor<1x3x1x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [3, 1, 1, 1], dim_mapping = [[0], [0], [1], [2, 3]]} : tensor<1x3x1x1xf16> -> tensor<3x1x1x1xf16>

  return %2 : tensor<3x1x1x1xf16>

  //CHECK: [[CONST:%.*]] =  const.Declare tensor<3x1x1x1x!qElemType0> = #const.Content<dense<1.000000e+00> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>, #const.Reshape<[3, 1, 1, 1]>]>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<3x1x1x1x!qElemType0> -> tensor<3x1x1x1xf16>
  //CHECK: return [[DEQUANT]] : tensor<3x1x1x1xf16>
}

// -----

!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType0 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagationDequantPerAxisReshapeOneToZeroAxis
func @PropagationDequantPerAxisReshapeOneToZeroAxis(%arg0: tensor<1x3x2x1x!qElemType0>) -> tensor<3x2x1x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x2x1x!qElemType0> -> tensor<1x3x2x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [3, 2, 1, 1], dim_mapping = [[0], [0], [1], [2, 3]]} : tensor<1x3x2x1xf16> -> tensor<3x2x1x1xf16>

  return %2 : tensor<3x2x1x1xf16>

  //CHECK: [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [3, 2, 1, 1]} : tensor<1x3x2x1x!qElemType0> -> tensor<3x2x1x1x!qElemType1>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[RESHAPE]]) {dstElemType = f16} : tensor<3x2x1x1x!qElemType1> -> tensor<3x2x1x1xf16>
  //CHECK: return [[DEQUANT]] : tensor<3x2x1x1xf16>
}

// -----

!qElemType = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @NoPropagationDequantPerAxisReshape
func @NoPropagationDequantPerAxisReshape(%arg0: tensor<1x3x1x2x!qElemType>) -> tensor<1x2x1x3xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  %2 = IE.Reshape(%1) {shape_value = [1, 2, 1, 3]} : tensor<1x3x1x2xf16> -> tensor<1x2x1x3xf16>

  return %2 : tensor<1x2x1x3xf16>

  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x1x2x!qElemType> -> tensor<1x3x1x2xf16>
  //CHECK: [[RESHAPE:%.*]] = IE.Reshape([[DEQUANT]]) {shape_value = [1, 2, 1, 3]} : tensor<1x3x1x2xf16> -> tensor<1x2x1x3xf16>
  //CHECK: return [[RESHAPE]] : tensor<1x2x1x3xf16>
}

// -----

!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagationDequantPerAxisReshapeZeroToOneAxis
func @PropagationDequantPerAxisReshapeZeroToOneAxis(%arg0: tensor<3x2x1x1x!qElemType0>) -> tensor<1x3x2x1xf16> {
  %1 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<3x2x1x1x!qElemType0> -> tensor<3x2x1x1xf16>
  %2 = IE.AffineReshape(%1) {shape_value = [1, 3, 2, 1], dim_mapping = [[0, 1], [2], [3], [3]]} : tensor<3x2x1x1xf16> -> tensor<1x3x2x1xf16>

  return %2 : tensor<1x3x2x1xf16>

  //CHECK: [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 3, 2, 1]} : tensor<3x2x1x1x!qElemType0> -> tensor<1x3x2x1x!qElemType1>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x2x1x!qElemType1> -> tensor<1x3x2x1xf16>
  //CHECK: return [[DEQUANT]] : tensor<1x3x2x1xf16>
}

// -----

!qElemType = type !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>

// CHECK-LABEL: @PropagateDequantConstPerAxisExpandDilated
func @PropagateDequantConstPerAxisExpandDilated() -> tensor<1x3x5x5xf16> {
  %0 = const.Declare tensor<1x3x3x3x!qElemType> = #const.Content<dense<1.0> : tensor<1x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x3x3x!qElemType> -> tensor<1x3x3x3xf16>
  %2 = IE.ExpandDilated(%1) {dilations = [2, 2]} : tensor<1x3x3x3xf16> -> tensor<1x3x5x5xf16>

  return %2 : tensor<1x3x5x5xf16>

  //CHECK: [[CONST:%.*]] =  const.Declare tensor<1x3x5x5x!qElemType> = #const.Content<dense<1.000000e+00> : tensor<1x3x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>, #const.ExpandDilated<[2, 2]>]>
  //CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[CONST]]) {dstElemType = f16} : tensor<1x3x5x5x!qElemType> -> tensor<1x3x5x5xf16>
  //CHECK: return [[DEQUANT]] : tensor<1x3x5x5xf16>
}

// -----
