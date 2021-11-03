// RUN: vpux-opt --split-input-file --convert-quantize-ops-to-eltwise  %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8:f32, 2.000000e+00>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>
!qElemType2 = type !quant.uniform<u8:f32, 5.000000e-01>

func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x4xf32> -> tensor<1x4x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType1> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>

    // CHECK:  %[[VAL0:.*]] = IE.Add(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!qElemType0>
    // CHECK:  %[[VAL1:.*]] = IE.QuantizeCast(%[[VAL0]]) {dstElemType = !qElemType1} : tensor<1x4x!qElemType0> -> tensor<1x4x!qElemType1>
    // CHECK:  %[[VAL2:.*]] = IE.QuantizeCast(%[[VAL1]]) {dstElemType = !qElemType2} : tensor<1x4x!qElemType1> -> tensor<1x4x!qElemType2>
    // CHECK:  %[[VAL3:.*]] = IE.Add(%[[VAL2]], %[[VAL2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4x!qElemType2>, tensor<1x4x!qElemType2> -> tensor<1x4xf32>
    // CHECK:  return %[[VAL3]] : tensor<1x4xf32>
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
