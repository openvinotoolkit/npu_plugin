//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --sparsify-weights %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparsifyPaddedConv
func @SparsifyPaddedConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x4xsi32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x15x1x1xf16>, [
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
    %1 = VPU.NCE.Convolution(%arg0, %weights, %arg1) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[weights:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:    : tensor<16x15x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.Sparsify<false, dense<15> : tensor<16xi64>>]
    // CHECK:       [[sparsity_map:%.+]] = const.Declare tensor<16x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:    : tensor<16x15x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.GetSparsityMap]
    // CHECK:       [[new_weights:%.+]] = VPU.GroupSparseTensor([[weights]], [[sparsity_map]])
    // CHECK-SAME:      {compression_scheme = #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<15> : tensor<16xi64>, alignment = 16 : i64>, is_weights}
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                         #VPU.CompressionScheme<axis = 0 : i64, numElems = dense<15> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK:       VPU.NCE.Convolution(%arg0, [[new_weights]], %arg1)
}
