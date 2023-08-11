//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-exclude-pad-for-avg-pool %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @HandleExcludePadForAvgPool
func.func @HandleExcludePadForAvgPool(%arg0 : tensor<1x1024x7x7xf16>) -> (tensor<1x1024x7x7xf16>) {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x7xf16>
    return %0 : tensor<1x1024x7x7xf16>

    //CHECK:        [[VAR:%.+]] = IE.AvgPool(%arg0)
    //CHECK-SAME:   {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x7xf16> -> tensor<1x1024x5x5xf16>

    //CHECK:        [[VAR0:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 2, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR1:%.+]] = IE.AvgPool([[VAR0]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR2:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 0, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 2, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR3:%.+]] = IE.AvgPool([[VAR2]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR4:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 5, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR5:%.+]] = IE.AvgPool([[VAR4]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR6:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 5, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x2xf16>

    //CHECK:        [[VAR7:%.+]] = IE.AvgPool([[VAR6]])
    //CHECK-SAME:   {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x2xf16> -> tensor<1x1024x1x1xf16>

    //CHECK:        [[VAR8:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 2]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x2xf16>

    //CHECK:        [[VAR9:%.+]] = IE.AvgPool([[VAR8]])
    //CHECK-SAME:   {kernel_size = [3, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x2xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[VAR10:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 0, 5]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x7x2xf16>

    //CHECK:        [[VAR11:%.+]] = IE.AvgPool([[VAR10]])
    //CHECK-SAME:   {kernel_size = [3, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x7x2xf16> -> tensor<1x1024x5x1xf16>

    //CHECK:        [[VAR12:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 0, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 2, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x7xf16>

    //CHECK:        [[VAR13:%.+]] = IE.AvgPool([[VAR12]])
    //CHECK-SAME:   {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x7xf16> -> tensor<1x1024x1x5xf16>

    //CHECK:        [[VAR14:%.+]] = IE.StridedSlice(%arg0)
    //CHECK-SAME:   begins_attr = [0, 0, 5, 0]
    //CHECK-SAME:   ends_attr = [0, 0, 7, 7]
    //CHECK-SAME:   : tensor<1x1024x7x7xf16> -> tensor<1x1024x2x7xf16>

    //CHECK:        [[VAR15:%.+]] = IE.AvgPool([[VAR14]])
    //CHECK-SAME:   {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x7xf16> -> tensor<1x1024x1x5xf16>

    //CHECK:        [[VAR17:%.+]] = IE.Concat([[VAR]], [[VAR1]], [[VAR3]], [[VAR5]], [[VAR7]], [[VAR9]], [[VAR11]], [[VAR13]], [[VAR15]])
    //CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 6], [0, 0, 6, 6], [0, 0, 6, 0], [0, 0, 1, 0], [0, 0, 1, 6], [0, 0, 0, 1], [0, 0, 6, 1]]}
    //CHECK-SAME:   : tensor<1x1024x5x5xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x5x1xf16>, tensor<1x1024x1x5xf16>, tensor<1x1024x1x5xf16> -> tensor<1x1024x7x7xf16>

    // CHECK        return [[VAR17]]
}
