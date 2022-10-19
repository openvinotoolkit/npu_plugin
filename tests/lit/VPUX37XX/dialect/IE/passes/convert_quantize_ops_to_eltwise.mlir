// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-quantize-ops-to-nce-ops  %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8:f32, 2.000000e+00>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>
!qElemType2 = type !quant.uniform<u8:f32, 5.000000e-01>

module @ConvertQuantizeToEltwiseVPUX37XX {

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

!qElemType = type !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

module @ConvertQuantizeToDwConvVPUX37XX {

func @PerAxis(%arg0 : tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    // CHECK:  %[[VAL0:.*]] = const.Declare tensor<3x1x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<3x1x1x1xf16>>
    // CHECK:  %[[VAL1:.*]] = IE.GroupConvolution(%arg0, %[[VAL0]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 3 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x3x16x16xf16>, tensor<3x1x1x1xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK:  %[[VAL2:.*]] = IE.Dequantize(%[[VAL1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:  return %[[VAL2]] : tensor<1x3x16x16xf16>
}

}
