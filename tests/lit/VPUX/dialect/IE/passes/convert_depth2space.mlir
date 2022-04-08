//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-depthToSpace %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Depth2SpaceCannotConvertToNNDMAsWithLargeHeight_BLOCKS_FIRST
func @Depth2SpaceCannotConvertToNNDMAsWithLargeHeight_BLOCKS_FIRST(%arg0: tensor<1x4x512x8xf16>) -> tensor<1x1x1024x16xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>

    return %0 : tensor<1x1x1024x16xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 2, 2, 1, 512, 8]} : tensor<1x4x512x8xf16> -> tensor<1x2x2x1x512x8xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #map} : tensor<1x2x2x1x512x8xf16> -> tensor<1x1x512x2x8x2xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [1, 1, 1024, 16]} : tensor<1x1x512x2x8x2xf16> -> tensor<1x1x1024x16xf16>
    // CHECK:       return [[VAL2]] : tensor<1x1x1024x16xf16>
}
