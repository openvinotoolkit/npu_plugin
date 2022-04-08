//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-for-vpu %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

module @Test {

// CHECK-LABEL: @AdjustForVPU
func @AdjustForVPU(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x64xf16> {
    %cts = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1], pads_begin = [2], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64xf16> -> tensor<1x1x64xf16>

    return %1 : tensor<1x1x64xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>, [#const.Reshape<[1, 16, 1, 5]>]

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 16, 1, 64]} : tensor<1x16x64xf16> -> tensor<1x16x1x64xf16>

    // CHECK: [[VAL1:%.*]] = IE.Convolution([[VAL0]], [[CST]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 2],
    // CHECK-SAME:    strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x64xf16>

    // CHECK: [[VAL2:%.*]] = IE.AffineReshape([[VAL1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>

    // CHECK: [[VAL3:%.*]] = IE.ReLU([[VAL2]]) : tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK: return [[VAL3]] : tensor<1x1x64xf16>
}

// -----
// CHECK-LABEL: @AdjustForVPU1
func @AdjustForVPU1(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x64x64xf16> {
    %cts = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x64x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64x64xf16> -> tensor<1x1x64x64xf16>

    return %1 : tensor<1x1x64x64xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[CST]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:    post_op = {attrs = {}, name = "IE.ReLU"}, strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x64x64xf16>

    // CHECK: return [[VAL1]] : tensor<1x1x64x64xf16>
}

}
