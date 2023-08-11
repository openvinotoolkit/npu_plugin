//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --remove-identity-pools %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @RemoveIdentityAvgPool
func.func @RemoveIdentityAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK-NOT:   IE.AvgPool
}

// CHECK-LABEL: @RemoveIdentityMaxPool
func.func @RemoveIdentityMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK-NOT:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPool
func.func @NotRemoveIdentityAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x9x12xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x9x12xf16>

    return %ave_pool : tensor<1x64x9x12xf16>
    // CHECK:   IE.AvgPool
}

// CHECK-LABEL: @NotRemoveIdentityMaxPool
func.func @NotRemoveIdentityMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x9x12xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x9x12xf16>

    return %max_pool : tensor<1x64x9x12xf16>
    // CHECK:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPoolPostOp
func.func @NotRemoveIdentityAvgPoolPostOp(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = {attrs = {}, name = "IE.Sigmoid"},
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.AvgPool
}

// CHECK-LABEL: @NotRemoveIdentityMaxPoolPostOp
func.func @NotRemoveIdentityMaxPoolPostOp(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = {attrs = {}, name = "IE.Sigmoid"},
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.MaxPool
}

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>
// CHECK-LABEL: @NotRemoveIdentityMaxPoolDiffType
func.func @NotRemoveIdentityMaxPoolDiffType(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13x!qElemType>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13x!qElemType>

    return %max_pool : tensor<1x64x10x13x!qElemType>
    // CHECK:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPoolDiffType
func.func @NotRemoveIdentityAvgPoolDiffType(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13x!qElemType>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13x!qElemType>

    return %ave_pool : tensor<1x64x10x13x!qElemType>
    // CHECK:   IE.AvgPool
}
