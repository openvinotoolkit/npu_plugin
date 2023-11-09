//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-reflect-pad-to-slice-and-concat %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @conver2DPad
func.func @conver2DPad(%arg0: tensor<1x8x13x29xf16>) -> tensor<1x8x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x13x29xf16> -> tensor<1x8x16x32xf16>

    return %0 : tensor<1x8x16x32xf16>

    // CHEAK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 8, 1, 29] : tensor<1x8x13x29xf16> to tensor<1x8x1x29xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 0, 11, 0] [1, 8, 1, 29] : tensor<1x8x13x29xf16> to tensor<1x8x1x29xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice %arg0 [0, 0, 10, 0] [1, 8, 1, 29] : tensor<1x8x13x29xf16> to tensor<1x8x1x29xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[SLICE0]], %arg0, [[SLICE1]], [[SLICE2]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x8x1x29xf16>, tensor<1x8x13x29xf16>, tensor<1x8x1x29xf16>, tensor<1x8x1x29xf16> -> tensor<1x8x16x29xf16>
    // CHECK:       [[SLICE3:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 0, 2] [1, 8, 16, 1] : tensor<1x8x16x29xf16> to tensor<1x8x16x1xf16>
    // CHECK:       [[SLICE4:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 0, 1] [1, 8, 16, 1] : tensor<1x8x16x29xf16> to tensor<1x8x16x1xf16>
    // CHECK:       [[SLICE5:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 0, 27] [1, 8, 16, 1] : tensor<1x8x16x29xf16> to tensor<1x8x16x1xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[SLICE3]], [[SLICE4]], [[CONCAT0]], [[SLICE5]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x8x16x1xf16>, tensor<1x8x16x1xf16>, tensor<1x8x16x29xf16>, tensor<1x8x16x1xf16> -> tensor<1x8x16x32xf16>
    // CHECK:       return [[CONCAT1]] : tensor<1x8x16x32xf16>
}

// CHECK-LABEL: @fuse2DPadwith3DType
func.func @fuse2DPadwith3DType(%arg0: tensor<8x13x29xf16>) -> tensor<8x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 1, 2], pads_end_attr = [0, 2, 1]}
                        : tensor<8x13x29xf16> -> tensor<8x16x32xf16>

    return %0 : tensor<8x16x32xf16>

    // CHEAK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 1, 0] [8, 1, 29] : tensor<8x13x29xf16> to tensor<8x1x29xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 11, 0] [8, 1, 29] : tensor<8x13x29xf16> to tensor<8x1x29xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice %arg0 [0, 10, 0] [8, 1, 29] : tensor<8x13x29xf16> to tensor<8x1x29xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[SLICE0]], %arg0, [[SLICE1]], [[SLICE2]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<8x1x29xf16>, tensor<8x13x29xf16>, tensor<8x1x29xf16>, tensor<8x1x29xf16> -> tensor<8x16x29xf16>
    // CHECK:       [[SLICE3:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 2] [8, 16, 1] : tensor<8x16x29xf16> to tensor<8x16x1xf16>
    // CHECK:       [[SLICE4:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 1] [8, 16, 1] : tensor<8x16x29xf16> to tensor<8x16x1xf16>
    // CHECK:       [[SLICE5:%.*]] = IE.Slice [[CONCAT0]] [0, 0, 27] [8, 16, 1] : tensor<8x16x29xf16> to tensor<8x16x1xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[SLICE3]], [[SLICE4]], [[CONCAT0]], [[SLICE5]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<8x16x1xf16>, tensor<8x16x1xf16>, tensor<8x16x29xf16>, tensor<8x16x1xf16> -> tensor<8x16x32xf16>
    // CHECK:       return [[CONCAT1]] : tensor<8x16x32xf16>
}

// CHECK-LABEL: @convert4DPad
func.func @convert4DPad(%arg0: tensor<16x8x13x29xf16>) -> tensor<19x11x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [2, 1, 1, 2], pads_end_attr = [1, 2, 2, 1]}
                        : tensor<16x8x13x29xf16> -> tensor<19x11x16x32xf16>

    return %0 : tensor<19x11x16x32xf16>

    // CHEAK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [2, 0, 0, 0] [1, 8, 13, 29] : tensor<16x8x13x29xf16> to tensor<1x8x13x29xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 8, 13, 29] : tensor<16x8x13x29xf16> to tensor<1x8x13x29xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice %arg0 [14, 0, 0, 0] [1, 8, 13, 29] : tensor<16x8x13x29xf16> to tensor<1x8x13x29xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[SLICE0]], [[SLICE1]], %arg0, [[SLICE2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x8x13x29xf16>, tensor<1x8x13x29xf16>, tensor<16x8x13x29xf16>, tensor<1x8x13x29xf16> -> tensor<19x8x13x29xf16>
    // CHECK:       [[SLICE3:%.*]] = IE.Slice [[CONCAT0]] [0, 1, 0, 0] [19, 1, 13, 29] : tensor<19x8x13x29xf16> to tensor<19x1x13x29xf16>
    // CHECK:       [[SLICE4:%.*]] = IE.Slice [[CONCAT0]] [0, 6, 0, 0] [19, 1, 13, 29] : tensor<19x8x13x29xf16> to tensor<19x1x13x29xf16>
    // CHECK:       [[SLICE5:%.*]] = IE.Slice [[CONCAT0]] [0, 5, 0, 0] [19, 1, 13, 29] : tensor<19x8x13x29xf16> to tensor<19x1x13x29xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<19x1x13x29xf16>, tensor<19x8x13x29xf16>, tensor<19x1x13x29xf16>, tensor<19x1x13x29xf16> -> tensor<19x11x13x29xf16>
    // CHECK:       [[SLICE6:%.*]] = IE.Slice [[CONCAT1]] [0, 0, 1, 0] [19, 11, 1, 29] : tensor<19x11x13x29xf16> to tensor<19x11x1x29xf16>
    // CHECK:       [[SLICE7:%.*]] = IE.Slice [[CONCAT1]] [0, 0, 11, 0] [19, 11, 1, 29] : tensor<19x11x13x29xf16> to tensor<19x11x1x29xf16>
    // CHECK:       [[SLICE8:%.*]] = IE.Slice [[CONCAT1]] [0, 0, 10, 0] [19, 11, 1, 29] : tensor<19x11x13x29xf16> to tensor<19x11x1x29xf16>
    // CHECK:       [[CONCAT2:%.*]] = IE.Concat([[SLICE6]], [[CONCAT1]], [[SLICE7]], [[SLICE8]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<19x11x1x29xf16>, tensor<19x11x13x29xf16>, tensor<19x11x1x29xf16>, tensor<19x11x1x29xf16> -> tensor<19x11x16x29xf16>
    // CHECK:       [[SLICE9:%.*]] = IE.Slice [[CONCAT2]] [0, 0, 0, 2] [19, 11, 16, 1] : tensor<19x11x16x29xf16> to tensor<19x11x16x1xf16>
    // CHECK:       [[SLICE10:%.*]] = IE.Slice [[CONCAT2]] [0, 0, 0, 1] [19, 11, 16, 1] : tensor<19x11x16x29xf16> to tensor<19x11x16x1xf16>
    // CHECK:       [[SLICE11:%.*]] = IE.Slice [[CONCAT2]] [0, 0, 0, 27] [19, 11, 16, 1] : tensor<19x11x16x29xf16> to tensor<19x11x16x1xf16>
    // CHECK:       [[CONCAT3:%.*]] = IE.Concat([[SLICE9]], [[SLICE10]], [[CONCAT2]], [[SLICE11]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<19x11x16x1xf16>, tensor<19x11x16x1xf16>, tensor<19x11x16x29xf16>, tensor<19x11x16x1xf16> -> tensor<19x11x16x32xf16>
    // CHECK:       return [[CONCAT3]] : tensor<19x11x16x32xf16>
}
