//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --initial-transformations="convert-fc-to-conv=true" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @TransformPassesWithFC
func.func @TransformPassesWithFC(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK-DAG:       [[WEIGHTS:%.*]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.*]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 1, 1]} : tensor<1x16xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       [[VAL1:%.*]] = IE.Reshape([[WEIGHTS]]) {shape_value = [64, 16, 1, 1]} : tensor<64x16xf32> -> tensor<64x16x1x1xf32>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[BIAS]]) {shape_value = [1, 64, 1, 1]} : tensor<1x64xf32> -> tensor<1x64x1x1xf32>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[VAL0]], [[VAL1]], [[VAL2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       [[VAL3:%.*]] = IE.Reshape([[CONV]]) {shape_value = [1, 64]} : tensor<1x64x1x1xf32> -> tensor<1x64xf32>
    // CHECK:       return [[VAL3]]
}


// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>


    // CHECK-DAG:      [[CST_0:%.*]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>, #const.Transpose<#map>]
    // CHECK-DAG:      [[CST_1:%.*]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>, #const.Transpose<#map>]
    // CHECK:          [[IN_1:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_1_2D:%.*]] = IE.AffineReshape([[IN_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:          [[IN_2:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:          [[IN_2_2D:%.*]] = IE.AffineReshape([[IN_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:          [[IN_1_4D:%.*]] = IE.Reshape([[IN_1_2D]]) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    // CHECK:          [[WEIGHTS_1:%.*]] = IE.Reshape([[CST_1]]) {shape_value = [40, 512, 1, 1]} : tensor<40x512xf32> -> tensor<40x512x1x1xf32>

    // CHECK:          [[CONV_1:%.*]] = IE.Convolution([[IN_1_4D]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf32>, tensor<40x512x1x1xf32> -> tensor<1x40x1x1xf32>
    // CHECK:          [[OUT_1_2D:%.*]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 40]} : tensor<1x40x1x1xf32> -> tensor<1x40xf32>

    // CHECK:          [[IN_2_4D:%.*]] = IE.Reshape([[IN_2_2D]]) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    // CHECK:          [[WEIGHTS_2:%.*]] = IE.Reshape([[CST_0]]) {shape_value = [40, 512, 1, 1]} : tensor<40x512xf32> -> tensor<40x512x1x1xf32>
    // CHECK:          [[CONV_2:%.*]] = IE.Convolution([[IN_2_4D]], [[WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf32>, tensor<40x512x1x1xf32> -> tensor<1x40x1x1xf32>
    // CHECK:          [[OUT_2_2D:%.*]] = IE.Reshape([[CONV_2]]) {shape_value = [1, 40]} : tensor<1x40x1x1xf32> -> tensor<1x40xf32>

    // CHECK:          [[CONCAT:%.*]] = IE.Concat([[OUT_1_2D]], [[OUT_2_2D]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0], [1, 0]]} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:          [[OUT:%.*]] = IE.AffineReshape([[CONCAT]])

    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 40]} : tensor<2x40xf32> -> tensor<1x2x1x40xf32>
    // CHECK return [[OUT]] : tensor<1x2x1x40xf32>
}
