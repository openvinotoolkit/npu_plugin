//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --enable-weights-sparsity="weights-sparsity-heuristic=RATIO" %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module attributes {VPU.arch = "VPUX37XX"} {

// CHECK-LABEL: @DoNotSparsifyCompressedConv
func @DoNotSparsifyCompressedConv(%arg0: tensor<1x4x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x4xsi32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.0> : tensor<16x3x1x1xf16>, [
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.Reshape<[16, 1, 1, 4]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    %1 = VPU.NCE.Convolution(%arg0, %weights, %arg1) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 4, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   const.Sparsify
    // CHECK-NOT:   const.GetSparsityMap
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK:       [[weights:%.+]] = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:    : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>,
    // CHECK-SAME:                             #const.Reshape<[16, 1, 1, 4]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK:       VPU.NCE.Convolution(%arg0, [[weights]], %arg1)
}

}
