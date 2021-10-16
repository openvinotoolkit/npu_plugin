// RUN: vpux-opt --split-input-file --convert-quantize-ops-to-eltwise  %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f32, 1.000000e+00>
func @PerTensor(%arg0 : tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4xf32> -> tensor<1x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<1x4x!qElemType> -> tensor<1x4xf32>

    // CHECK:  %0 = IE.And(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4xf32>, tensor<1x4xf32> -> tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00>>
    // CHECK:  %1 = IE.And(%0, %0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00>> -> tensor<1x4xf32>

    return %1 : tensor<1x4xf32>
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
