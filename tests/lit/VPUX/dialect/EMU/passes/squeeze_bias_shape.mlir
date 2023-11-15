//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --squeeze-bias-shape %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @SqueezeBiasShapeConv
func.func @SqueezeBiasShapeConv(%arg0: tensor<1x16x64x4xf16>) -> tensor<1x32x64x4xf16> {
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.ConvertElemType<f16>]
    %weights = const.Declare tensor<32x16x3x1xf16> = dense<1.0> : tensor<32x16x3x1xf32>, [#const.ConvertElemType<f16>]
    %0 = VPU.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1], groups = 1 : i64} : tensor<1x16x64x4xf16>, tensor<32x16x3x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x64x4xf16>
    return %0 : tensor<1x32x64x4xf16>

    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<32xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[32]>]
    // CHECK-DAG:   %[[WEIGHTS:.*]] = const.Declare tensor<32x16x3x1xf16>

    // CHECK:       %[[CONV0:.*]] = VPU.GroupConvolution(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK:       return  %[[CONV0]]
}

// -----

// CHECK-LABEL: @SqueezeBiasShapeFC
func.func @SqueezeBiasShapeFC(%arg0: tensor<1x16xf16>) -> tensor<1x32xf16> {
    %bias = const.Declare tensor<1x32xf16> = dense<1.0> : tensor<1x32xf32>, [#const.ConvertElemType<f16>]
    %weights = const.Declare tensor<32x16xf16> = dense<1.0> : tensor<32x16xf32>, [#const.ConvertElemType<f16>]
    %0 = VPU.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf16>, tensor<32x16xf16>, tensor<1x32xf16> -> tensor<1x32xf16>
    return %0 : tensor<1x32xf16>

    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<32xf16> = dense<1.000000e+00> : tensor<1x32xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[32]>]
    // CHECK-DAG:   %[[WEIGHTS:.*]] = const.Declare tensor<32x16xf16>

    // CHECK:       %[[CONV0:.*]] = VPU.FullyConnected(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK:       return  %[[CONV0]]
}
