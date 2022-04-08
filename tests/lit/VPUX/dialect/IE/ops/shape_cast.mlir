//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Eliminate
func @Eliminate(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16>
    return %0 : tensor<1x2x3x4xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       return %arg0
}

// CHECK-LABEL: @Fuse
func @Fuse(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16> {
    %0 = IE.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    %1 = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x2x3x4xf16>
    return %1 : tensor<1x2x3x4xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK:       return %arg0
}

// CHECK-LABEL: @FuseSequence
func @FuseSequence(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x4x3x2xf16> {
    %0 = IE.ShapeCast {shape = [1, 3, 2, 4]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x3x2x4xf16>
    %1 = IE.ShapeCast {shape = [1, 3, 4, 2]} inputs(%0 : tensor<1x3x2x4xf16>) -> tensor<1x3x4x2xf16>
    %2 = IE.ShapeCast {shape = [1, 4, 3, 2]} inputs(%1 : tensor<1x3x4x2xf16>) -> tensor<1x4x3x2xf16>
    return %2 : tensor<1x4x3x2xf16>

    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 3, 2]} inputs(%arg0 : tensor<1x2x3x4xf16>) -> tensor<1x4x3x2xf16>
    // CHECK:       return [[SHAPE_CAST]]
}
