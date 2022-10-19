// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --fuse-quantized-ops %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = type !quant.uniform<u8:f16, 0.39320635328105852:128>
!qElemType2 = type !quant.uniform<u8:f16, 0.39320638320025275:128>

// CHECK-LABEL: @FuseQParamsIntoAddWithDiffInTypes
func @FuseQParamsIntoAddWithDiffInTypes(%arg0: tensor<1x16x180x320xf16>, %arg1: tensor<1x16x180x320xf16>) -> tensor<1x16x180x320xf16> {
  %0 = IE.Quantize(%arg0) {dstElemType = !qElemType0} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType0>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x180x320x!qElemType0> -> tensor<1x16x180x320xf16>

  %2 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>

  %4 = IE.Add(%1, %3) { auto_broadcast = "NUMPY" } : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x16x180x320xf16>

  %5 = IE.Quantize(%4) {dstElemType = !qElemType2} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType2>
  %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  return %6 : tensor<1x16x180x320xf16>

  //CHECK: [[VAL0:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType0} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType0>
  //CHECK: [[VAL1:%.*]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType0> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL2:%.*]] = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType1>
  //CHECK: [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType1> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL4:%.*]] = IE.Add([[VAL1]], [[VAL3]]) {auto_broadcast = "NUMPY"} : tensor<1x16x180x320xf16>, tensor<1x16x180x320xf16> -> tensor<1x16x180x320xf16>
  //CHECK: [[VAL5:%.*]] = IE.Quantize([[VAL4]]) {dstElemType = !qElemType2} : tensor<1x16x180x320xf16> -> tensor<1x16x180x320x!qElemType2>
  //CHECK: [[VAL6:%.*]] = IE.Dequantize([[VAL5]]) {dstElemType = f16} : tensor<1x16x180x320x!qElemType2> -> tensor<1x16x180x320xf16>
  //CHECK: return [[VAL6]] : tensor<1x16x180x320xf16>
}
