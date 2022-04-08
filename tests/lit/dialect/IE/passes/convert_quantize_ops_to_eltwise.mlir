// RUN: vpux-opt --split-input-file --convert-quantize-ops-to-eltwise  %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8:f32, 2.000000e+00>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>
!qElemType2 = type !quant.uniform<u8:f32, 5.000000e-01>

module @ConvertQuantizeToEltwiseVPUX37XX attributes {VPU.arch = "VPUX37XX"} {

func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x4xf32> -> tensor<1x4x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType1> -> tensor<1x4xf32>

    // CHECK:  %[[VAL0:.*]] = IE.Add(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!qElemType0>
    // CHECK:  %[[VAL1:.*]] = IE.QuantizeCast(%[[VAL0]]) {dstElemType = !qElemType1} : tensor<1x4x!qElemType0> -> tensor<1x4x!qElemType1>
    // CHECK:  %[[VAL2:.*]] = IE.QuantizeCast(%[[VAL1]]) {dstElemType = !qElemType2} : tensor<1x4x!qElemType1> -> tensor<1x4x!qElemType2>
    // CHECK:  %[[VAL3:.*]] = IE.Add(%[[VAL2]], %[[VAL2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4x!qElemType2>, tensor<1x4x!qElemType2> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>

    // CHECK:  return %[[VAL3]] : tensor<1x4xf32>
}

}

// -----

!qElemType = type !quant.uniform<u8:f32, 1.000000e+00>

module @ConvertQuantizeToEltwiseKMB attributes {VPU.arch = "VPUX30XX"} {

func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>

    // CHECK:  %[[VAL0:.*]] = IE.And(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!qElemType>
    // CHECK:  %[[VAL1:.*]] = IE.And(%[[VAL0]], %[[VAL0]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4x!qElemType>, tensor<1x4x!qElemType> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>

    // CHECK:  return %[[VAL1]] : tensor<1x4xf32>
}

}

// -----

!qElemType = type !quant.uniform<u8:f32:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

func @PerChannelNoChanges(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>

    // CHECK:  IE.Quantize
    // CHECK:  IE.Dequantize
    // CHECK-NOT:  IE.And
    // CHECK-NOT:  IE.And

    return %1 : tensor<1x4xf32>
}

// -----

!qElemType = type !quant.uniform<u8:f32, 1.000000e+00>

module @NoConvertQuantizeToEltwiseKMB attributes {VPU.arch = "VPUX30XX"} {

func @EnabelCMcovNoChanges(%arg0 : tensor<1x3x352x352xf32>) -> tensor<1x32x175x175x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x352x352xf32> -> tensor<1x3x352x352x!qElemType>
    %1 = const.Declare tensor<32x3x3x3xf32> = #const.Content<dense<1.0> : tensor<32x3x3x3xf32>>
    %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} :
                    tensor<1x3x352x352x!qElemType>, tensor<32x3x3x3xf32> -> tensor<1x32x175x175x!qElemType>

    return %2 : tensor<1x32x175x175x!qElemType>

    // CHECK:  [[QUANT0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x352x352xf32> -> tensor<1x3x352x352x!qElemType>
    // CHECK:  [[CST0:%.*]] = const.Declare tensor<32x3x3x3xf32> = #const.Content<dense<1.000000e+00> : tensor<32x3x3x3xf32>>
    // CHECK:  [[VAR0:%.*]] = IE.Convolution([[QUANT0]], [[CST0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]}
    // CHECK-SAME:  tensor<1x3x352x352x!qElemType>, tensor<32x3x3x3xf32> -> tensor<1x32x175x175x!qElemType>
    // CHECK:  return [[VAR0]] : tensor<1x32x175x175x!qElemType>
}

}
