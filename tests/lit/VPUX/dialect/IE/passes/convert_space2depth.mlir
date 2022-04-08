//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-spaceToDepth %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d5, d1, d2, d4)>

// CHECK-LABEL: @convertSpaceToDepth_BLOCKS_FIRST
func @convertSpaceToDepth_BLOCKS_FIRST(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x48x128x128xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = "BLOCKS_FIRST"} : tensor<1x3x512x512xf16> -> tensor<1x48x128x128xf16>

    return %0 : tensor<1x48x128x128xf16>

    //CHECK-NOT: IE.SpaceToDepthOp
    //CHECK: [[RESHAPE0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 3, 128, 4, 128, 4]} : tensor<1x3x512x512xf16> -> tensor<1x3x128x4x128x4xf16>
    //CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map} : tensor<1x3x128x4x128x4xf16> -> tensor<1x4x4x3x128x128xf16>
    //CHECK: [[RESHAPE1:%.*]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 48, 128, 128]} : tensor<1x4x4x3x128x128xf16> -> tensor<1x48x128x128xf16>
    //CHECK: return [[RESHAPE1]] : tensor<1x48x128x128xf16>
}
// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2, d4)>

// CHECK-LABEL: @convertSpaceToDepth_DEPTH_FIRST
func @convertSpaceToDepth_DEPTH_FIRST(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x48x128x128xf16> {
    %0 = IE.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x3x512x512xf16> -> tensor<1x48x128x128xf16>

    return %0 : tensor<1x48x128x128xf16>

    //CHECK-NOT: IE.SpaceToDepthOp
    //CHECK: [[RESHAPE0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 3, 128, 4, 128, 4]} : tensor<1x3x512x512xf16> -> tensor<1x3x128x4x128x4xf16>
    //CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map} : tensor<1x3x128x4x128x4xf16> -> tensor<1x3x4x4x128x128xf16>
    //CHECK: [[RESHAPE1:%.*]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [1, 48, 128, 128]} : tensor<1x3x4x4x128x128xf16> -> tensor<1x48x128x128xf16>
    //CHECK: return [[RESHAPE1]] : tensor<1x48x128x128xf16>
}
