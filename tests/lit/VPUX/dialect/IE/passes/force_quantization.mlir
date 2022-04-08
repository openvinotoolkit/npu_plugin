//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --force-host-quantization %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType = type !quant.uniform<u8:f16, 1.1534313725490195:128>

module @Quantize1 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf32>
    }
    outputsInfo : {
        DataInfo "output" : tensor<1x2x3x4xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x2x3x4x!qElemType>)
func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>

    return %2 : tensor<1x2x3x4xf16>

    // CHECK: %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    // CHECK: %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
    // CHECK: return %1 : tensor<1x2x3x4xf16>
}

}
