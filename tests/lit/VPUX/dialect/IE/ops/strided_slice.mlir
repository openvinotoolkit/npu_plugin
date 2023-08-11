//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x10x10x30xf16> {
    %begins = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends = const.Declare tensor<4xsi64> = dense<[1, 5, 10, 20]> : tensor<4xsi64>
    %strides = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 1]> : tensor<4xsi64>

    %0 = IE.StridedSlice(%arg0, %begins, %ends, %strides) {
        begin_mask = [0, 1, 1, 0],
        end_mask = [0, 1, 0, 1],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<1> : vector<4xi32>
    } : tensor<1x10x20x30xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x10x10x30xf16>

    return %0 : tensor<1x10x10x30xf16>
    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)
    // CHECK-SAME:  begins_attr = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 10, 20]
    // CHECK-SAME:  strides_attr = [1, 1, 1, 1]
}

// -----

//CHECK-LABEL: @NoComposeOnDifferentStrides
func.func @NoComposeOnDifferentStrides(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x3x320x320xf16> {
    %begins1 = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends1 = const.Declare tensor<4xsi64> = dense<[0, 0, 2147483647, 0]> : tensor<4xsi64>
    %strides1 = const.Declare tensor<4xsi64> = dense<[1, 1, 2, 1]> : tensor<4xsi64>
    %begins2 = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 0]> : tensor<4xsi64>
    %ends2 = const.Declare tensor<4xsi64> = dense<[0, 0, 0, 2147483647]> : tensor<4xsi64>
    %strides2 = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 2]> : tensor<4xsi64>

    %0 = IE.StridedSlice(%arg0, %begins1, %ends1, %strides1) {
        begin_mask = [1, 1, 1, 1],
        end_mask = [1, 1, 0, 1],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<1> : vector<4xi32>
    } : tensor<1x3x640x640xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x320x640xf16>
    %1 = IE.StridedSlice(%0, %begins2, %ends2, %strides2) {
        begin_mask = [1, 1, 1, 1],
        end_mask = [1, 1, 1, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<1> : vector<4xi32>
    } : tensor<1x3x320x640xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x3x320x320xf16>

    return %1 : tensor<1x3x320x320xf16>
    // CHECK:       [[VAL0:%.*]] = IE.StridedSlice(%arg0)
    // CHECK:       [[VAL1:%.*]] = IE.StridedSlice([[VAL0]])
}
