//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-to-mixed-precision %s | FileCheck %s

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

func.func @MixedPrecisionConvForOutputShape(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  return %4 : tensor<1x16x3x3xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

  //CHECK: [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x3x3xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>

// TODO: #67754
func.func @MixedPrecisionMaxPoolForOutputShape(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
  %3 = IE.MaxPool(%2) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
  return %3 : tensor<1x3x16x16xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0)
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
  //CHECK: [[VAL2:%.*]] = IE.MaxPool([[VAL1]])
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionGroupConvForOutputShape(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<16x1x1x1x!qElemType> = dense<2.000000e+00> : tensor<16x1x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x1x1x!qElemType> -> tensor<16x1x1x1xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<16x1x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionAvgPoolForOutputShape(%arg0: tensor<1x64x88x88x!qElemType>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x64x88x88x!qElemType> -> tensor<1x64x88x88xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x64x88x88xf16> -> tensor<1x64x11x11xf16>
  
    return %1 : tensor<1x64x11x11xf16>

    // CHECK-NOT: IE.Dequantize
    // CHECK:       %[[AVGPOOL:.*]] = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x64x88x88x!qElemType> -> tensor<1x64x11x11xf16>

}


// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionAdd(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.*]] = const.Declare tensor<1x16x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>]

    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Add([[VAL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.500000e+00>

func.func @MixedPrecisionAddForDifferentScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType0> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType0> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x16x3x3x!qElemType0> =
    // CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]

    // CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} :
    // CHECK-SAME:  tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>

    // CHECK: [[VAL3:%.*]] = IE.Add([[VAL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:  tensor<1x16x3x3x!qElemType1>, tensor<1x16x3x3x!qElemType0> -> tensor<1x16x3x3xf16>

    // CHECK: return [[VAL3]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.0025215686274509803>
!qElemType1 = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @FloatInConv2d(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType0> {
    %cst = const.Declare tensor<16x16x1x1x!qElemType1> = dense<2.000000e+00> : tensor<16x16x1x1xf16>, [
        #const.ConvertElemType<ui8>,
        #const.QuantCast<!qElemType1>
    ]

    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType0
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType0>

    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType0>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x3x3x!qElemType0>

    return %1 : tensor<1x16x3x3x!qElemType0>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]

    // CHECK:   [[DEQUANT:%.*]] = IE.Dequantize([[CST]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[DEQUANT]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType0>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType0>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.0025215686274509803>
!qElemType1 = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @FloatInGroupConv(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType0> {
    %cst = const.Declare tensor<16x1x1x1x!qElemType1> = dense<2.000000e+00> : tensor<16x1x1x1xf16>, [
        #const.ConvertElemType<ui8>,
        #const.QuantCast<!qElemType1>
    ]

    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType0
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType0>

    %1 = IE.GroupConvolution(%0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType0>, tensor<16x1x1x1x!qElemType1> -> tensor<1x16x3x3x!qElemType0>

    return %1 : tensor<1x16x3x3x!qElemType0>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x1x1x1x!qElemType1> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x1x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]

    // CHECK:   [[DEQUANT:%.*]] = IE.Dequantize([[CST]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x1x1x1x!qElemType1> -> tensor<16x1x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.GroupConvolution(%arg0, [[DEQUANT]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType0>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType0>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @FloatInAdd(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.Add(%0, %0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.Add(%arg0, %arg0) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @FloatInAvgPool(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.AvgPool(%0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @SkipFloatInMaxPool(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.MaxPool(%0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[QUANT:%.*]] = IE.Quantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.MaxPool([[QUANT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

func.func @DoNotConvertConvWithPReLU(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %1 = IE.Quantize(%arg0) {
      dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %2 = IE.Dequantize(%1) {
      dstElemType = f16
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [
        #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType>
    ]

    %3 = IE.Dequantize(%WEIGHTS) {
        dstElemType = f16
    } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    %4 = IE.Convolution(%2, %3) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        post_op = #IE.PostOp<
            name = "IE.LeakyRelu",
            attrs = {
                negative_slope = 2.500000e-01 : f64
            }
        >
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    return %4 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[WEIGHTS:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
    // CHECK-SAME:      #const.ConvertElemType<ui8>,
    // CHECK-SAME:      #const.QuantCast<!qElemType>
    // CHECK-SAME:  ]

    // CHECK: [[QUANT:%.*]] = IE.Quantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK: [[DEQUANT:%.*]] = IE.Dequantize([[QUANT]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[DEQUANT_WEIGHTS:%.*]] = IE.Dequantize([[WEIGHTS]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution([[DEQUANT]], [[DEQUANT_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.PostOp<
    // CHECK-SAME:          name = "IE.LeakyRelu",
    // CHECK-SAME:          attrs = {
    // CHECK-SAME:              negative_slope = 2.500000e-01 : f64
    // CHECK-SAME:          }
    // CHECK-SAME:      >,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

func.func @Conv2dWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

func.func @DoNotConv2dWithQuantize(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x5x5x!qElemType> {
    %cst = const.Declare tensor<16x16x12x12xf16> = dense<2.000000e+00> : tensor<16x16x12x12xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x16xf16>, tensor<16x16x12x12xf16> -> tensor<1x16x5x5xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x5x5xf16> -> tensor<1x16x5x5x!qElemType>

    return %1 : tensor<1x16x5x5x!qElemType>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x16x12x12xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x12x12xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x16x16xf16>, tensor<16x16x12x12xf16> -> tensor<1x16x5x5xf16>

    // CHECK:   [[VAL1:%.*]] = IE.Quantize([[VAL0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x5x5xf16> -> tensor<1x16x5x5x!qElemType>

    // CHECK:   return [[VAL1]] : tensor<1x16x5x5x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

func.func @GroupConvWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

    %0 = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x1x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.GroupConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

func.func @AvgPoolWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

func.func @DoNotAvgPoolWithQuantize(%arg0: tensor<1x1x64x128xf16>) -> tensor<1x1x1x128x!qElemType> {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [64, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x1x64x128xf16> -> tensor<1x1x1x128xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x1x1x128xf16> -> tensor<1x1x1x128x!qElemType>

    return %1 : tensor<1x1x1x128x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [64, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x1x64x128xf16> -> tensor<1x1x1x128xf16>
    //
    // CHECK: [[VAL1:%.*]] = IE.Quantize([[VAL0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      } : tensor<1x1x1x128xf16> -> tensor<1x1x1x128x!qElemType>
    //
    // CHECK:   return [[VAL1]] : tensor<1x1x1x128x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @AddWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Add(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.Add(%arg0, %arg0) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionGroupConvForOutputShapeWithQuantWeightsBias(%arg0: tensor<1x3x320x480xf16>) -> tensor<1x3x320x480xf16> {
    %cst = const.Declare tensor<1x3x1x1x!qElemType0> = dense<1.270000e+02> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]
    %cst_0 = const.Declare tensor<3x1x1x1x!qElemType1> = dense<2.000000e+00> : tensor<3x1x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x320x480xf16> -> tensor<1x3x320x480x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x320x480x!qElemType1> -> tensor<1x3x320x480xf16>
    %2 = IE.Dequantize(%cst_0) {dstElemType = f16} : tensor<3x1x1x1x!qElemType1> -> tensor<3x1x1x1xf16>
    %3 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x3x1x1x!qElemType0> -> tensor<1x3x1x1xf16>
    %4 = IE.GroupConvolution(%1, %2, %3) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x480xf16>, tensor<3x1x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x320x480xf16>
    return %4 : tensor<1x3x320x480xf16>

    //CHECK: [[CST0:%.*]] = const.Declare tensor<1x3x1x1x!qElemType0> =
    //CHECK-SAME:     dense<1.270000e+02> : tensor<1x3x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]
    //CHECK: [[CST:%.*]] = const.Declare tensor<3x1x1x1x!qElemType1> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<3x1x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType1>]
    //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x320x480xf16> -> tensor<1x3x320x480x!qElemType1>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[CST0]]) {dstElemType = f16} : tensor<1x3x1x1x!qElemType0> -> tensor<1x3x1x1xf16>
    //CHECK: [[VAL2:%.*]] = IE.GroupConvolution([[VAL0]], [[CST]], [[VAL1]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x480x!qElemType1>, tensor<3x1x1x1x!qElemType1>, tensor<1x3x1x1xf16> -> tensor<1x3x320x480xf16>
    //CHECK: return [[VAL2]] : tensor<1x3x320x480xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

func.func @DoNotAddWithQuantize(%arg0: tensor<1x8x8x32xf16>) -> tensor<1x8x8x32x!qElemType> {
    %cst0 = const.Declare tensor<1x1x1x32xf16> = dense<1.000000e+00> : tensor<1x1x1x32xf16>
    %0 = IE.Add(%arg0, %cst0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x8x8x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x8x8x32xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x8x8x32xf16> -> tensor<1x8x8x32x!qElemType>

    return %1 : tensor<1x8x8x32x!qElemType>

    // CHECK:   [[CST0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.000000e+00> : tensor<1x1x1x32xf16>
    // CHECK:   [[VAL0:%.*]] = IE.Add(%arg0, [[CST0]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x8x8x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x8x8x32xf16>
    //
    // CHECK:   [[VAL1:%.*]] = IE.Quantize([[VAL0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x8x8x32xf16> -> tensor<1x8x8x32x!qElemType>

    // CHECK:   return [[VAL1]] : tensor<1x8x8x32x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

func.func @Conv2dLeakyReluWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x3x3x3x!qElemType> {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> : tensor<3x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<
            name = "IE.LeakyRelu",
            attrs = {
                negative_slope = 2.500000e-01 : f64
            }
        >,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>

    return %1 : tensor<1x3x3x3x!qElemType>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<3x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.PostOp<
    // CHECK-SAME:          name = "IE.LeakyRelu",
    // CHECK-SAME:          attrs = {
    // CHECK-SAME:              negative_slope = 2.500000e-01 : f64
    // CHECK-SAME:          }
    // CHECK-SAME:      >,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x3x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.01:128,0.02:128,0.03:128}>

func.func @DoNotConv2dLeakyReluWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x3x3x3x!qElemType> {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> : tensor<3x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.PostOp<
            name = "IE.LeakyRelu",
            attrs = {
                negative_slope = 2.500000e-01 : f64
            }
        >,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>

    return %1 : tensor<1x3x3x3x!qElemType>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<3x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.PostOp<
    // CHECK-SAME:          name = "IE.LeakyRelu",
    // CHECK-SAME:          attrs = {
    // CHECK-SAME:              negative_slope = 2.500000e-01 : f64
    // CHECK-SAME:          }
    // CHECK-SAME:      >,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    // CHECK:  [[VAL1:%.*]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>
    // CHECK:   return [[VAL1]] : tensor<1x3x3x3x!qElemType>
}
