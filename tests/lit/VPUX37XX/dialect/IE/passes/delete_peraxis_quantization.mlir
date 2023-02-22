//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --delete-peraxis-quantization  %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f32:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

module @DeleteQuantizationPerAxis {

func @main(%arg0 : tensor<1x3x7x7xf32>) -> tensor<1x3x7x7xf32> {
    %0 = IE.HSwish(%arg0) : tensor<1x3x7x7xf32> -> tensor<1x3x7x7xf32>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x7x7xf32> -> tensor<1x3x7x7x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f32} : tensor<1x3x7x7x!qElemType> -> tensor<1x3x7x7xf32>

    // CHECK:     %[[VAL0:.*]] = IE.HSwish(%arg0) : tensor<1x3x7x7xf32> -> tensor<1x3x7x7xf32>
    // CHECK-NOT:  %[[VAL1:.*]] = IE.Quantize
    // CHECK-NOT:  %[[VAL2:.*]] = IE.Dequantize

    return %2 : tensor<1x3x7x7xf32>

    // CHECK:  return %[[VAL0]] : tensor<1x3x7x7xf32>
}

}

// -----

!qElemType = type !quant.uniform<u8:f32:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128}>

module @NoDeleteQuantizationPerAxisFusedPostOp {

func @main(%arg0 : tensor<1x3x7x7xf16>) -> tensor<1x3x7x7xf16> {
    %filter = const.Declare tensor<3x3x1x1xf16> = dense<1.0> : tensor<3x3x1x1xf16>
    %0 = IE.Convolution(%arg0, %filter)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1],
            post_op = {attrs = {}, name = "IE.ReLU"}
        } :
        tensor<1x3x7x7xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x7x7xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x7x7xf16> -> tensor<1x3x7x7x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x7x7x!qElemType> -> tensor<1x3x7x7xf16>

    // CHECK:     IE.Convolution
    // CHECK:     IE.Quantize
    // CHECK:     %[[RESULT:.*]] = IE.Dequantize

    return %2 : tensor<1x3x7x7xf16>

    // CHECK:  return %[[RESULT]] : tensor<1x3x7x7xf16>
}

}
