//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --adjust-for-optimized-sw-kernel %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @AdjustForSoftmaxAxisZeroOptNCHW
func.func @AdjustForSoftmaxAxisZeroOptNCHW(%arg0: tensor<1x64x16x1xf16>) -> tensor<1x64x16x1xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x1xf16> -> tensor<1x64x16x1xf16>
    return %0 : tensor<1x64x16x1xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 64, 1, 16]} inputs(%arg0 : tensor<1x64x16x1xf16>) -> tensor<1x64x1x16xf16>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x64x1x16xf16> -> tensor<1x64x1x16xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 64, 16, 1]} inputs([[SOFTMAX]] : tensor<1x64x1x16xf16>) -> tensor<1x64x16x1xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxAxisZeroOptNCHW
func.func @NotAdjustForSoftmaxAxisZeroOptNCHW(%arg0: tensor<1x64x16x16xf16>) -> tensor<1x64x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    return %0 : tensor<1x64x16x16xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x64x16x16xf16> -> tensor<1x64x16x16xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @AdjustForSoftmaxAxisZeroOptNHWC
func.func @AdjustForSoftmaxAxisZeroOptNHWC(%arg0: tensor<1x1x16x64xf16, {order = #NHWC}>) -> tensor<1x1x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x1x16x64xf16, {order = #NHWC}> -> tensor<1x1x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x1x16x64xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 64, 16, 1]} inputs(%arg0 : tensor<1x1x16x64xf16, {order = #NHWC}>) -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x64x16x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 16, 64]} inputs([[SOFTMAX]] : tensor<1x64x16x1xf16, {order = #NHWC}>) -> tensor<1x1x16x64xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxAxisZeroOptNHWC
func.func @NotAdjustForSoftmaxAxisZeroOptNHWC(%arg0: tensor<1x2x16x64xf16, {order = #NHWC}>) -> tensor<1x2x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @AdjustForSoftmaxMultiShaveOptNCHW
func.func @AdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 32, 1, 32]} inputs(%arg0 : tensor<1x2x16x32xf16>) -> tensor<1x32x1x32xf16>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x32x1x32xf16> -> tensor<1x32x1x32xf16>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[SOFTMAX]] : tensor<1x32x1x32xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNCHW
func.func @NotAdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    return %0 : tensor<1x2x16x16xf16>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x2x16x16xf16> -> tensor<1x2x16x16xf16>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// CHECK-LABEL:   @AdjustForSoftmaxMultiShaveOptNHWC
func.func @AdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 32, 32, 1]} inputs(%arg0 : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x32x1xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x32x32x1xf16, {order = #NHWC}> -> tensor<1x32x32x1xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[SOFTMAX]] : tensor<1x32x32x1xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// CHECK-LABEL:   @NotAdjustForSoftmaxMultiShaveOptNHWC
func.func @NotAdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x2x16x64xf16, {order = #NHWC}>) -> tensor<1x2x16x64xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x2x16x64xf16, {order = #NHWC}> -> tensor<1x2x16x64xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}
