//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scatterndupdate-to-strided-concat
// --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// For example, if there is a 1x15 tensor: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
// The Indices to update data is [0,3,6,9,12] . The data to update is [fx0,fx1,fx2,fx3,fx4]
// The results is [fx0, 2, 3, fx1, 5, 6, fx2, 8, 9, fx3, 11, 12, fx4, 14, 15].
// It equals to offset 0, stride 3, strided concat.
// The cst is the indices values. For ConvertScatterNDUpdateToStridedConcat test, it match the condition and could be converted.

// CHECK-LABEL: @ConvertScatterNDUpdateToStridedConcat
func.func @ConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,12]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_1:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[SLICE_2:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 2], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE_1]], [[SLICE_2]]) {per_axis = {axis = 4 : i64, offset = 1 : i64, stride = 3 : i64}} : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[CONCAT]] : tensor<1x1x1x1x15xf16>
}

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

// The input and output shape are exactly the same. Do not convert.
// CHECK-LABEL: @NoProperStrideDoNotConvertScatterNDUpdateToStridedConcat
func.func @NoProperStrideDoNotConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x5xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x5xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,0,3],[0,0,0,0,4]]]]]]>  : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x5xf16>

    return %0 : tensor<1x1x1x1x5xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1x5x5xsi32> =
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate(%arg0, [[CST]], %arg1) : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: return [[RESULT]] : tensor<1x1x1x1x5xf16>
}
