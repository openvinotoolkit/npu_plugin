//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat-slice %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @NoChangesAcrossInputsAtHeight
func.func @NoChangesAcrossInputsAtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

    // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL}    {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} :
    // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK:        [[VAR1:%.+]] = IE.Slice [[VAR0]]
    // CHECK-SAME:          [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    // CHECK: return [[VAR1]] : tensor<1x16x3x4xf16>
}

// CHECK-LABEL: @ChangesInInput0AtHeight
func.func @ChangesInInput0AtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x4x4xf16> to tensor<1x16x3x4xf16>
   // CHECK: return [[VAR0]] : tensor<1x16x3x4xf16>
}

// CHECK-LABEL: @ChangesInInput1SameShapeAtHeight
func.func @ChangesInInput1SameShapeAtHeight(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: return %arg1 : tensor<1x16x3x4xf16>
}

// CHECK-LABEL: @ChangesAtHeightTwoUsers
func.func @ChangesAtHeightTwoUsers(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> (tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    %2 = IE.Slice %0 [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1, %2 : tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} :
   // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
   // CHECK: [[VAR1:%.+]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 3, 4] : tensor<1x16x4x4xf16> to tensor<1x16x3x4xf16>
   // CHECK: [[VAR2:%.+]] = IE.Slice [[VAR0]]
   // CHECK-SAME:           [0, 0, 2, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
   // CHECK:  return [[VAR1]], [[VAR2:%.+]] : tensor<1x16x3x4xf16>, tensor<1x16x3x4xf16>

}

// CHECK-LABEL: @NoChangesNoOffsetAttribute
func.func @NoChangesNoOffsetAttribute(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1) { per_axis = #IE.Concat<axis = 2 : i64> } : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    %1 = IE.Slice %0 [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

    // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16> -> tensor<1x16x7x4xf16>
    // CHECK: [[VAR1:%.+]] = IE.Slice [[VAR0]] [0, 0, 4, 0] [1, 16, 3, 4] : tensor<1x16x7x4xf16> to tensor<1x16x3x4xf16>
    // CHECK: return [[VAR1:%.+]] : tensor<1x16x3x4xf16>

}

// CHECK-LABEL: @NoChangesAtHeightWidth
func.func @NoChangesAtHeightWidth(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>, %arg2 : tensor<1x16x7x3xf16>) -> tensor<1x16x3x4xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
    %1 = IE.Slice %0 [0, 0, 2, 2] [1, 16, 3, 4] : tensor<1x16x7x7xf16> to tensor<1x16x3x4xf16>
    return %1 : tensor<1x16x3x4xf16>

   // CHECK: [[VAR0:%.+]] = IE.Concat(%arg0, %arg1, %arg2)
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} :
   // CHECK-SAME:            tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
   // CHECK: [[VAR1:%.+]] = IE.Slice [[VAR0]]
   // CHECK-SAME:           [0, 0, 2, 2] [1, 16, 3, 4] : tensor<1x16x7x7xf16> to tensor<1x16x3x4xf16>
   // CHECK:  return [[VAR1:%.+]] : tensor<1x16x3x4xf16>

}

// CHECK-LABEL: @ChangesAtHeightWidth
func.func @ChangesAtHeightWidth(%arg0: tensor<1x16x4x4xf16>, %arg1 : tensor<1x16x3x4xf16>, %arg2 : tensor<1x16x7x3xf16>) -> tensor<1x16x2x2xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]} : tensor<1x16x4x4xf16>, tensor<1x16x3x4xf16>, tensor<1x16x7x3xf16> -> tensor<1x16x7x7xf16>
    %1 = IE.Slice %0 [0, 0, 5, 1] [1, 16, 2, 2] : tensor<1x16x7x7xf16> to tensor<1x16x2x2xf16>
    return %1 : tensor<1x16x2x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Slice %arg1 [0, 0, 1, 1] [1, 16, 2, 2] : tensor<1x16x3x4xf16> to tensor<1x16x2x2xf16>
    // CHECK: return [[VAR0]] : tensor<1x16x2x2xf16>

}

// CHECK-LABEL: @ChangesAtChannelHeight
func.func @ChangesAtChannelHeight(%arg0: tensor<1x3x4x4xf16>, %arg1 : tensor<1x3x3x4xf16>, %arg2 : tensor<1x2x7x4xf16>) -> tensor<1x2x2x2xf16> {
    %0 = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 3, 0, 0]]} : tensor<1x3x4x4xf16>, tensor<1x3x3x4xf16>, tensor<1x2x7x4xf16> -> tensor<1x5x7x4xf16>
    %1 = IE.Slice %0 [0, 1, 5, 1] [1, 2, 2, 2] : tensor<1x5x7x4xf16> to tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK: [[VAR0:%.+]] = IE.Slice %arg1 [0, 1, 1, 1] [1, 2, 2, 2] : tensor<1x3x3x4xf16> to tensor<1x2x2x2xf16>
    // CHECK: return [[VAR0]] : tensor<1x2x2x2xf16>

}


