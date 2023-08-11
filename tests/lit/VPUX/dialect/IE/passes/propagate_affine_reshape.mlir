//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-affine-reshape --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @ShiftReshapeAndTranspose(%arg0: tensor<1x12x7x1xf32>, %arg1: tensor<1x1x12x7xf32>) -> (tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x12x7x1xf32> -> tensor<12x7x1x1xf32>
    %1 = IE.Transpose(%0) {order_value = #CNHW} : tensor<12x7x1x1xf32> -> tensor<7x12x1x1xf32>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %3 = IE.Transpose(%2) {order_value = #CNHW} : tensor<12x7x1x1xf32> -> tensor<7x12x1x1xf32>

    return %1, %3 : tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x12x7x1xf32> -> tensor<1x7x12x1xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [7, 12, 1, 1]} : tensor<1x7x12x1xf32> -> tensor<7x12x1x1xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg1) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [7, 12, 1, 1]} : tensor<1x1x7x12xf32> -> tensor<7x12x1x1xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
#WHC = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

func.func @CollapseReshapeAndTranspose(%arg0: tensor<1x1x12x7xf32>) -> (tensor<7x12xf32>, tensor<7x1x12xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 7] } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<12x7xf32> -> tensor<7x12xf32>

    %2 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 1, 7] } : tensor<1x1x12x7xf32> -> tensor<12x1x7xf32>
    %3 = IE.Transpose(%2) {order_value = #WHC} : tensor<12x1x7xf32> -> tensor<7x1x12xf32>

    return %1, %3 : tensor<7x12xf32>, tensor<7x1x12xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [7, 12]} : tensor<1x1x7x12xf32> -> tensor<7x12xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [7, 1, 12]} : tensor<1x1x7x12xf32> -> tensor<7x1x12xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<7x12xf32>, tensor<7x1x12xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @ExtendReshapeAndTranspose(%arg0: tensor<7x12xf32>) -> (tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 7, 12] } : tensor<7x12xf32> -> tensor<1x1x7x12xf32>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x1x7x12xf32> -> tensor<1x1x12x7xf32>

    %2 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1]} : tensor<7x12xf32> -> tensor<1x7x12x1xf32>
    %3 = IE.Transpose(%2) {order_value = #NHCW} : tensor<1x7x12x1xf32> -> tensor<1x12x7x1xf32>

    return %1, %3 : tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<7x12xf32> -> tensor<12x7xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 12, 7]} : tensor<12x7xf32> -> tensor<1x1x12x7xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<7x12xf32> -> tensor<12x7xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 12, 7, 1]} : tensor<12x7xf32> -> tensor<1x12x7x1xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>
}

// -----

func.func @ShiftReshapeAndExpand(%arg0: tensor<1x1x12x7xf32>, %arg1: tensor<1x19x80x1xf16>) -> (tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 6, 0, 0]} : tensor<12x7x1x1xf32> -> tensor<12x16x1x1xf32>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 19, 80]} : tensor<1x19x80x1xf16> -> tensor<1x1x19x80xf16>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x19x80xf16> -> tensor<1x16x19x80xf16>

    return %1, %3 : tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>

    // CHECK:               [[EXPAND0:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 3], pads_end = [0, 0, 0, 6]} : tensor<1x1x12x7xf32> -> tensor<1x1x12x16xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[EXPAND0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 16, 1, 1]} : tensor<1x1x12x16xf32> -> tensor<12x16x1x1xf32>

    // CHECK:               [[EXPAND1:%.*]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [15, 0, 0, 0]} : tensor<1x19x80x1xf16> -> tensor<16x19x80x1xf16>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[EXPAND1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 19, 80]} : tensor<16x19x80x1xf16> -> tensor<1x16x19x80xf16>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>
}

// -----

func.func @CollapseReshapeAndExpand(%arg0: tensor<1x1x12x7xf32>) -> tensor<16x7xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 7] } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 0], pads_end = [4, 0]}  : tensor<12x7xf32> -> tensor<16x7xf32>
    return %1 : tensor<16x7xf32>

    // CHECK:               [[EXPAND:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x1x12x7xf32> -> tensor<1x1x16x7xf32>
    // CHECK:               [[RESHAPE:%.*]]  = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [16, 7]} : tensor<1x1x16x7xf32> -> tensor<16x7xf32>
    // CHECK:               return [[RESHAPE]] : tensor<16x7xf32>
}

// -----

func.func @ExtendReshapeAndExpand(%arg0: tensor<7x12xf32>) -> tensor<1x7x16x1xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1] } : tensor<7x12xf32> -> tensor<1x7x12x1xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 2, 0], pads_end = [0, 0, 2, 0]} : tensor<1x7x12x1xf32> -> tensor<1x7x16x1xf32>
    return %1 : tensor<1x7x16x1xf32>

    // CHECK:               [[EXPAND:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 2], pads_end = [0, 2]} : tensor<7x12xf32> -> tensor<7x16xf32>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 16, 1]} : tensor<7x16xf32> -> tensor<1x7x16x1xf32>
    // CHECK:               return [[RESHAPE]] : tensor<1x7x16x1xf32>
}

// -----

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

func.func @NoChangesShiftReshape(%arg0: tensor<1x1x12x8xf32>) -> tensor<4x12x2x1xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 4, 2, 1]} : tensor<1x1x12x8xf32> -> tensor<12x4x2x1xf32>
    %1 = IE.Transpose(%0) {order_value = #CNHW} : tensor<12x4x2x1xf32> -> tensor<4x12x2x1xf32>

    return %1 : tensor<4x12x2x1xf32>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 4, 2, 1]} : tensor<1x1x12x8xf32> -> tensor<12x4x2x1xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE]]) {order_value = #map} : tensor<12x4x2x1xf32> -> tensor<4x12x2x1xf32>
    // CHECK:               return [[TRANSPOSE]] : tensor<4x12x2x1xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @NoChangesCollapseReshape(%arg0: tensor<1x2x12x7xf32>, %arg1: tensor<1x1x12x7xf32>, %arg2: tensor<1x1x12x6xf32>)
                                    -> (tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [24, 7] } : tensor<1x2x12x7xf32> -> tensor<24x7xf32>
    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<24x7xf32> -> tensor<7x24xf32>

    %2 = IE.AffineReshape(%arg1) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 7, 1] } : tensor<1x1x12x7xf32> -> tensor<12x7x1xf32>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0], pads_end = [0, 0, 15]}  : tensor<12x7x1xf32> -> tensor<12x7x16xf32>

    %4 = IE.AffineReshape(%arg2) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2] } : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]}  : tensor<12x3x2xf32> -> tensor<12x3x16xf32>

    return %1, %3, %5 : tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [24, 7]} : tensor<1x2x12x7xf32> -> tensor<24x7xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #map} : tensor<24x7xf32> -> tensor<7x24xf32>


    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 7, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1xf32>
    // CHECK:               [[EXPAND0:%.*]] = IE.Expand([[RESHAPE1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 15]} : tensor<12x7x1xf32> -> tensor<12x7x16xf32>

    // CHECK:               [[RESHAPE2:%.*]] = IE.AffineReshape(%arg2)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2]} : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
    // CHECK:               [[EXPAND1:%.*]] = IE.Expand([[RESHAPE2]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<12x3x2xf32> -> tensor<12x3x16xf32>

    // CHECK:               return [[TRANSPOSE]], [[EXPAND0]], [[EXPAND1]] : tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @NoChangesExpandReshape(%arg0: tensor<6x12xf32>) -> tensor<1x2x12x3xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 3, 12] } : tensor<6x12xf32> -> tensor<1x2x3x12xf32>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x2x3x12xf32> -> tensor<1x2x12x3xf32>
    return %1 : tensor<1x2x12x3xf32>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 3, 12]} : tensor<6x12xf32> -> tensor<1x2x3x12xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x2x3x12xf32> -> tensor<1x2x12x3xf32>
    // CHECK:               return [[TRANSPOSE]] : tensor<1x2x12x3xf32>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeWithConcatWithOffsets
func.func @SwapAffineReshapeWithConcatWithOffsets(%arg0: tensor<1x1024x1x1xf16>, %arg1: tensor<1x1024x1x1xf16>) ->
                        tensor<2x1x1024xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1024xf16>

      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1024xf16>

     %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [1, 0, 0]]} : tensor<1x1x1024xf16>, tensor<1x1x1024xf16> -> tensor<2x1x1024xf16>

     return %2 : tensor<2x1x1024xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<2x1024x1x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [2, 1, 1024]}
     // CHECK-SAME:  tensor<2x1024x1x1xf16> -> tensor<2x1x1024xf16>
}

func.func @SwapAffineReshapeWithConcatWithAxis(%arg0: tensor<8x76x1x1xf16>, %arg1: tensor<8x76x1x1xf16>, %arg2: tensor<8x76x1x1xf16>, %arg3: tensor<8x76x1x1xf16>) ->
                        tensor<32x76xf16> {

     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 76]} : tensor<8x76x1x1xf16> -> tensor<8x76xf16>
     %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 76]} : tensor<8x76x1x1xf16> -> tensor<8x76xf16>
     %2 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 76]} : tensor<8x76x1x1xf16> -> tensor<8x76xf16>
     %3 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 76]} : tensor<8x76x1x1xf16> -> tensor<8x76xf16>
     %4 = IE.Concat(%0, %1, %2, %3) {per_axis = {axis = 0}} : tensor<8x76xf16>, tensor<8x76xf16>, tensor<8x76xf16>, tensor<8x76xf16> -> tensor<32x76xf16>

     return %4 : tensor<32x76xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [24, 0, 0, 0]]}
     // CHECK-SAME:           tensor<8x76x1x1xf16>, tensor<8x76x1x1xf16>, tensor<8x76x1x1xf16>, tensor<8x76x1x1xf16> -> tensor<32x76x1x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [1]], shape_value = [32, 76]}
     // CHECK-SAME:  tensor<32x76x1x1xf16> -> tensor<32x76xf16>
}

// -----

// CHECK-LABEL: @NoSwapAffineReshapeConcatDiffMapping
func.func @NoSwapAffineReshapeConcatDiffMapping(%arg0: tensor<1x70x1x28xf16>, %arg1: tensor<1x70x28x1xf16>) ->
                        tensor<1x70x56xf16> {

      %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 70, 28]} : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 70, 28]} : tensor<1x70x28x1xf16> -> tensor<1x70x28xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [0, 0, 28]]} : tensor<1x70x28xf16>, tensor<1x70x28xf16> -> tensor<1x70x56xf16>

      return %2 : tensor<1x70x56xf16>

     // CHECK:       IE.AffineReshape(%arg0)
     // CHECK:       IE.AffineReshape(%arg1)
     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0], [0, 0, 28]]}
     // CHECK-SAME:           tensor<1x70x28xf16>, tensor<1x70x28xf16> -> tensor<1x70x56xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameDims
func.func @SwapAffineReshapeConcatSameDims(%arg0: tensor<1x1x12x7xf16>, %arg1: tensor<1x1x12x7xf16>) ->
                        tensor<24x7x1x1xf16> {

      %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf16> -> tensor<12x7x1x1xf16>
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf16> -> tensor<12x7x1x1xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [12, 0, 0, 0]]} : tensor<12x7x1x1xf16>, tensor<12x7x1x1xf16> -> tensor<24x7x1x1xf16>

      return %2 : tensor<24x7x1x1xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]}
     // CHECK-SAME:           tensor<1x1x12x7xf16>, tensor<1x1x12x7xf16> -> tensor<1x1x24x7xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [24, 7, 1, 1]}
     // CHECK-SAME:  tensor<1x1x24x7xf16> -> tensor<24x7x1x1xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameDims
func.func @SwapAffineReshapeConcatSameDims(%arg0: tensor<7x12xf16>, %arg1: tensor<7x12xf16>) ->
                        tensor<1x7x24x1xf16> {

      %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1] } : tensor<7x12xf16> -> tensor<1x7x12x1xf16>
      %1 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1] } : tensor<7x12xf16> -> tensor<1x7x12x1xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]} : tensor<1x7x12x1xf16>, tensor<1x7x12x1xf16> -> tensor<1x7x24x1xf16>

      return %2 : tensor<1x7x24x1xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [0, 12]]}
     // CHECK-SAME:           tensor<7x12xf16>, tensor<7x12xf16> -> tensor<7x24xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 24, 1]}
     // CHECK-SAME:  tensor<7x24xf16> -> tensor<1x7x24x1xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameMappingDims
func.func @SwapAffineReshapeConcatSameMappingDims(%arg0: tensor<1x256x1x1xf16>, %arg1: tensor<1x256x1x1xf16>) ->
                        tensor<1x1x256x2xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 256, 1]} : tensor<1x256x1x1xf16> -> tensor<1x1x256x1xf16>

     %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 256, 1]} : tensor<1x256x1x1xf16> -> tensor<1x1x256x1xf16>

     %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x1x256x1xf16>, tensor<1x1x256x1xf16> -> tensor<1x1x256x2xf16>

     return %2 : tensor<1x1x256x2xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x2x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 256, 2]}
     // CHECK-SAME:  tensor<1x256x2x1xf16> -> tensor<1x1x256x2xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameMappingDims
func.func @SwapAffineReshapeConcatSameMappingDims(%arg0: tensor<1x256x1x1xf16>, %arg1: tensor<1x256x1x1xf16>) ->
                        tensor<1x1x1x512xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256x1x1xf16> -> tensor<1x1x1x256xf16>

     %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256x1x1xf16> -> tensor<1x1x1x256xf16>

     %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x1x1x512xf16>

     return %2 : tensor<1x1x1x512xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x512x1x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 512]}
     // CHECK-SAME:  tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameMappingDims
func.func @SwapAffineReshapeConcatSameMappingDims(%arg0: tensor<3x2x1x1x1xf16>, %arg1: tensor<3x2x1x1x1xf16>) ->
                        tensor<1x3x4x2xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 3, 2, 1]} : tensor<3x2x1x1x1xf16> -> tensor<1x3x2x1xf16>

     %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 3, 2, 1]} : tensor<3x2x1x1x1xf16> -> tensor<1x3x2x1xf16>

     %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 1]]} : tensor<1x3x2x1xf16>, tensor<1x3x2x1xf16> -> tensor<1x3x4x2xf16>

     return %2 : tensor<1x3x4x2xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0, 0], [0, 2, 0, 1, 0]]} : tensor<3x2x1x1x1xf16>, tensor<3x2x1x1x1xf16> -> tensor<3x4x1x2x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [2], [3], [3]], shape_value = [1, 3, 4, 2]}
     // CHECK-SAME:  tensor<3x4x1x2x1xf16> -> tensor<1x3x4x2xf16>
}

// -----

// CHECK-LABEL: @SwapMultiAffineReshapeConcatSameMappingDims
func.func @SwapMultiAffineReshapeConcatSameMappingDims(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>,  %arg2: tensor<1x1x1x1xf16>) ->
                        tensor<1x3xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1]} : tensor<1x1x1x1xf16> -> tensor<1x1xf16>

     %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1]} : tensor<1x1x1x1xf16> -> tensor<1x1xf16>

     %2 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 1]} : tensor<1x1x1x1xf16> -> tensor<1x1xf16>

     %3 = IE.Concat(%0, %1, %2) {static_offsets = [[0, 0], [0, 1], [0, 2]]} : tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16> -> tensor<1x3xf16>

     return %3 : tensor<1x3xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 3]}
     // CHECK-SAME:  tensor<1x3x1x1xf16> -> tensor<1x3xf16>
}

// -----

// CHECK-LABEL: @NoSwapAffineReshapeConcatAxis
func.func @NoSwapAffineReshapeConcatAxis(%arg0: tensor<1x1x12x6xf32>, %arg1: tensor<1x1x12x6xf32>) ->
                        tensor<12x3x4xf32> {

      %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2] } : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
      %1 = IE.AffineReshape(%arg1) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2] } : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>

      %2 = IE.Concat(%0, %1) {per_axis = {axis = 2}} : tensor<12x3x2xf32>, tensor<12x3x2xf32> -> tensor<12x3x4xf32>

      return %2 : tensor<12x3x4xf32>

     // CHECK:       IE.AffineReshape(%arg0)
     // CHECK:       IE.AffineReshape(%arg1)
     // CHECK:       IE.Concat
     // CHECK-SAME:  tensor<12x3x2xf32>, tensor<12x3x2xf32> -> tensor<12x3x4xf32>
}

// -----

// CHECK-LABEL: @NoPropagateAfiineReshapeDifferentOutput
func.func @NoPropagateAfiineReshapeDifferentOutput(%arg0: tensor<1x32x32x16xf16>, %arg1: tensor<1x16x16x20xf16>) -> tensor<1x21504xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 16384]} : tensor<1x32x32x16xf16> -> tensor<1x16384xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 5120]} : tensor<1x16x16x20xf16> -> tensor<1x5120xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0], [0, 16384]]} : tensor<1x16384xf16>, tensor<1x5120xf16> -> tensor<1x21504xf16>

    return %2 : tensor<1x21504xf16>
    // CHECK: IE.AffineReshape
    // CHECK: IE.AffineReshape
    // CHECK: IE.Concat
    // CHECK-SAME: tensor<1x16384xf16>, tensor<1x5120xf16> -> tensor<1x21504xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatOnNewOneDim
func.func @SwapAffineReshapeConcatOnNewOneDim(%arg0: tensor<1x68x128x128xf16>, %arg1: tensor<1x68x128x128xf16>) ->
                        tensor<1x4x68x128x128xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %4 = IE.Concat(%0, %1, %2, %3) {static_offsets = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 3, 0, 0, 0]]} : tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16> -> tensor<1x4x68x128x128xf16>

    return %4 :tensor<1x4x68x128x128xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} : tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16> -> tensor<4x68x128x128xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 4, 68, 128, 128]}
     // CHECK-SAME:  tensor<4x68x128x128xf16> -> tensor<1x4x68x128x128xf16>
}
