//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-transpose --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithSoftmax
func.func @SwapWithSoftmax(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.SoftMax(%1) {axisInd = -1 : i64} : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[TRANSPOSE:%.*]] = IE.Transpose([[SOFTMAX]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK:        return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithGelu
func.func @SwapWithGelu(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.Gelu(%1) : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK: [[GELU:%.*]] = IE.Gelu(%arg0) : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[GELU]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}
