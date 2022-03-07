// RUN: vpux-opt --split-input-file --fuse-convert-with-quantize %s | FileCheck %s

// CHECK: !qElemType = type !quant.uniform<u8:f16, 1.000000e+00>
!qElemType = type !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @PerTensor
func @PerTensor(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} :
    //CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    //CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK: !qElemType = type !quant.uniform<u8:f16:1, {1.000000e+00,1.000000e+00,1.000000e+00}>
!qElemType = type !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxis
func @PerAxis(%arg0: tensor<1x3x16x16xui8>) -> tensor<1x3x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x16x16xui8> -> tensor<1x3x16x16xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %2 : tensor<1x3x16x16xf16>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} :
    //CHECK-SAME:   tensor<1x3x16x16xui8> -> tensor<1x3x16x16x!qElemType>
    //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    //CHECK: return [[VAL1]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = type !quant.uniform<u8:f16, 1.0:128>

// CHECK-LABEL: @PerTensorDequantizeConvert
func @PerTensorDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = ui8} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    //CHECK: return [[VAL0]] : tensor<1x3x16x16xui8>
}


// -----

!qElemType = type !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>

// CHECK-LABEL: @PerAxisDequantizeConvert
func @PerAxisDequantizeConvert(%arg0: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xui8> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %1 = IE.Convert(%0) {dstElemType = ui8} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xui8>

    return %1 : tensor<1x3x16x16xui8>

    //CHECK: [[VAL0:%.*]] = IE.QuantizeCast(%arg0) {dstElemType = ui8} :
    //CHECK-SAME:   tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xui8>

    //CHECK: return [[VAL0]] : tensor<1x3x16x16xui8>
}
