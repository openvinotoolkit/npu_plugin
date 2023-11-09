//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scatterndupdate-to-strided-concat
// --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertScatterNDUpdateToStridedConcat
func.func @ConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,12]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_1:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[SLICE_2:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 2], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE_1]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 4 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[CONCAT]] : tensor<1x1x1x1x15xf16>
}

// -----

// The last dim value is [0,0,0,0,11], so it will remain IE.ScatterNDUpdate.
// CHECK-LABEL: @DoNotConvertScatterNDUpdateToStridedConcat
func.func @DoNotConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,11]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1x5x5xsi32> =
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate(%arg0, [[CST]], %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[RESULT]] : tensor<1x1x1x1x15xf16>
}

// -----

// The indices shape could not meet Integer stride condition, so it will remain IE.ScatterNDUpdate.
// CHECK-LABEL: @NotIntegerStrideDoNotConvertScatterNDUpdateToStridedConcat
func.func @NotIntegerStrideDoNotConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x4x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9]]]]]]> : tensor<1x1x1x1x4x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x4x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1x4x5xsi32> =
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate(%arg0, [[CST]], %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x4x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[RESULT]] : tensor<1x1x1x1x15xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithSameSize
func.func @ConvertToSliceConcatElementsUpdateWithSameSize(%arg0:  tensor<1x1x1x1x5xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x5xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,0,3],[0,0,0,0,4]]]]]]>  : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x5xf16>

    return %0 : tensor<1x1x1x1x5xf16>

    // CHECK-NOT: ScatterNDUpdate
    // CHECK: return %arg1 : tensor<1x1x1x1x5xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithOneElement
func.func @ConvertToSliceConcatElementsUpdateWithOneElement(%arg0:  tensor<1x1x1x1xf16>, %arg1 : tensor<1x1x1x1xf16> ) -> tensor<1x1x1x1xf16>{
    %cst = const.Declare tensor<1x1x1x1x4xsi32> = dense<[[[[[0,0,0,0]]]]]>  : tensor<1x1x1x1x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1xf16>, tensor<1x1x1x1x4xsi32>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    return %0 : tensor<1x1x1x1xf16>

    // CHECK-NOT: ScatterNDUpdate
    // CHECK: return %arg1 : tensor<1x1x1x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithReplaceOneElement
func.func @ConvertToSliceConcatElementsUpdateWithReplaceOneElement(%arg0:  tensor<1x10x1xf16>, %arg1 : tensor<1x1x1xf16> ) -> tensor<1x10x1xf16>{
    %cst = const.Declare tensor<1x1x1x3xsi32> = dense<0> : tensor<1x1x1x3xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x10x1xf16>, tensor<1x1x1x3xsi32>, tensor<1x1x1xf16> -> tensor<1x10x1xf16>

    return %0 : tensor<1x10x1xf16>

    // CHECK: [[SLICE:%.*]] = IE.Slice %arg0 [0, 1, 0] [1, 9, 1] : tensor<1x10x1xf16> to tensor<1x9x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 1, 0]]} : tensor<1x1x1xf16>, tensor<1x9x1xf16> -> tensor<1x10x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x10x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithTwoSlice
func.func @ConvertToSliceConcatElementsUpdateWithTwoSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 249, 0]], [[0, 250, 0]], [[0, 251, 0]], [[0, 252, 0]], [[0, 253, 0]], [[0, 254, 0]], [[0, 255, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_LEFT:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 249, 1] : tensor<1x326x1xf16> to tensor<1x249x1xf16>
    // CHECK: [[SLICE_RIGHT:%.*]] = IE.Slice %arg0 [0, 256, 0] [1, 70, 1] : tensor<1x326x1xf16> to tensor<1x70x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_LEFT]], %arg1, [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 249, 0], [0, 256, 0]]} : tensor<1x249x1xf16>, tensor<1x7x1xf16>, tensor<1x70x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithRightSlice
func.func @ConvertToSliceConcatElementsUpdateWithRightSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 0, 0]], [[0, 1, 0]], [[0, 2, 0]], [[0, 3, 0]], [[0, 4, 0]], [[0, 5, 0]], [[0, 6, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_RIGHT:%.*]] = IE.Slice %arg0 [0, 7, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 7, 0]]} : tensor<1x7x1xf16>, tensor<1x319x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithLeftSlice
func.func @ConvertToSliceConcatElementsUpdateWithLeftSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 319, 0]], [[0, 320, 0]], [[0, 321, 0]], [[0, 322, 0]], [[0, 323, 0]], [[0, 324, 0]], [[0, 325, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_LEFT:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_LEFT]], %arg1) {static_offsets = {{\[\[}}0, 0, 0], [0, 319, 0]]} : tensor<1x319x1xf16>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @NotConvertToSliceConcatElementsUpdateWithIllegalIndicesData
func.func @NotConvertToSliceConcatElementsUpdateWithIllegalIndicesData(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 319, 0]], [[0, 320, 0]], [[0, 100, 0]], [[0, 322, 0]], [[0, 323, 0]], [[0, 324, 0]], [[0, 325, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.ConvertElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate
    // CHECK: return [[RESULT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatTensorUpdate
func.func @ConvertToSliceConcatTensorUpdate(%arg0:  tensor<4x4x4xf16>, %arg1 : tensor<2x4x4xf16> ) -> tensor<4x4x4xf16>{
    %cst = const.Declare tensor<2x1xsi32> = dense<[[1], [2]]> : tensor<2x1xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x4x4xf16>, tensor<2x1xsi32>, tensor<2x4x4xf16> -> tensor<4x4x4xf16>

    return %0 : tensor<4x4x4xf16>

    // CHECK: [[SLICE_LEFT:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 4, 4] : tensor<4x4x4xf16> to tensor<1x4x4xf16>
    // CHECK: [[SLICE_RIGHT:%.*]] = IE.Slice %arg0 [3, 0, 0] [1, 4, 4] : tensor<4x4x4xf16> to tensor<1x4x4xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_LEFT]], %arg1, [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [1, 0, 0], [3, 0, 0]]} : tensor<1x4x4xf16>, tensor<2x4x4xf16>, tensor<1x4x4xf16> -> tensor<4x4x4xf16>
    // CHECK: return [[CONCAT]] : tensor<4x4x4xf16>
}

// -----

// CHECK-LABEL: @NotConvertToSliceConcatTensorUpdateWithIllegalIndicesData
func.func @NotConvertToSliceConcatTensorUpdateWithIllegalIndicesData(%arg0:  tensor<4x4x4xf16>, %arg1 : tensor<2x4x4xf16> ) -> tensor<4x4x4xf16>{
    %cst = const.Declare tensor<2x1xsi32> = dense<[[2], [1]]> : tensor<2x1xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x4x4xf16>, tensor<2x1xsi32>, tensor<2x4x4xf16> -> tensor<4x4x4xf16>

    return %0 : tensor<4x4x4xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate
    // CHECK: return [[RESULT]] : tensor<4x4x4xf16>
}
