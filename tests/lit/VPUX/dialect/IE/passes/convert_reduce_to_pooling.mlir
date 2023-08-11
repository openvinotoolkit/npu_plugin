//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-reduce-to-pooling %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertReduceMeanToPooling4D
func.func @ConvertReduceMeanToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x1xf16>
  return %1 : tensor<1x1x1x1xf16>

  // CHECK:       %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 50], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPooling3D
func.func @ConvertReduceMeanToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<256x7x7xf16>, tensor<1xsi32> -> tensor<256x1x7xf16>
  return %1 : tensor<256x1x7xf16>

  // CHECK:       IE.Reshape(%arg0) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceMean
  // CHECK:       %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       %2 = IE.Reshape(%1) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOneKeepDim
func.func @ConvertReduceMeanToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMean(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x50xf16>
  return %1 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMeanToPoolingReduceDimOne
func.func @ConvertReduceMeanToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMean(%arg0, %cst) : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x50xf16>
  return %1 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMean
}

// CHECK-LABEL: @ConvertReduceMaxToPooling4D
func.func @ConvertReduceMaxToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x1xf16>
  return %1 : tensor<1x1x1x1xf16>
  // CHECK:       %0 = IE.MaxPool(%arg0) {kernel_size = [1, 50], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceMaxToPooling3D
func.func @ConvertReduceMaxToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<256x7x7xf16>, tensor<1xsi32> -> tensor<256x1x7xf16>
  return %1 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceMax
  // CHECK:       %1 = IE.MaxPool(%0) {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       %2 = IE.Reshape(%1) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOneKeepDim
func.func @ConvertReduceMaxToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x50xf16>
  return %1 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceMaxToPoolingReduceDimOne
func.func @ConvertReduceMaxToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMax(%arg0, %cst) : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x50xf16>
  return %1 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceMax
}

// CHECK-LABEL: @ConvertReduceSumToPooling4D
func.func @ConvertReduceSumToPooling4D(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x1xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x1xf16>
  return %1 : tensor<1x1x1x1xf16>

  // CHECK:       %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 50], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x1x50xf16> -> tensor<1x1x1x1xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK-DAG:       %cst_0 = const.Declare tensor<1xf16> = dense<5.000000e+01> : tensor<1xf16>
  // CHECK:       %1 = IE.Multiply(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1xf16> -> tensor<1x1x1x1xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPooling3D
func.func @ConvertReduceSumToPooling3D(%arg0: tensor<256x7x7xf16>) -> tensor<256x1x7xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<256x7x7xf16>, tensor<1xsi32> -> tensor<256x1x7xf16>
  return %1 : tensor<256x1x7xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 256, 7, 7]} : tensor<256x7x7xf16> -> tensor<1x256x7x7xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x256x7x7xf16> -> tensor<1x256x1x7xf16>
  // CHECK-DAG:       %cst_0 = const.Declare tensor<1xf16> = dense<7.000000e+00> : tensor<1xf16>
  // CHECK:       %2 = IE.Multiply(%1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x7xf16>, tensor<1xf16> -> tensor<1x256x1x7xf16>
  // CHECK:       %3 = IE.Reshape(%2) {shape_value = [256, 1, 7]} : tensor<1x256x1x7xf16> -> tensor<256x1x7xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOneKeepDim
func.func @ConvertReduceSumToPoolingReduceDimOneKeepDim(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x1x50xf16>
  return %1 : tensor<1x1x1x50xf16>

  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingReduceDimOne
func.func @ConvertReduceSumToPoolingReduceDimOne(%arg0: tensor<1x1x1x50xf16>) -> tensor<1x1x50xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceSum(%arg0, %cst) : tensor<1x1x1x50xf16>, tensor<1xsi32> -> tensor<1x1x50xf16>
  return %1 : tensor<1x1x50xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 50]} : tensor<1x1x1x50xf16> -> tensor<1x1x50xf16>
  // CHECK-NOT:   ReduceSum
}

// CHECK-LABEL: @ConvertReduceSumToPoolingAvoidingExpand
func.func @ConvertReduceSumToPoolingAvoidingExpand(%arg0: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x12x368x480xf16>, tensor<1xsi32> -> tensor<1x1x368x480xf16>
  return %1 : tensor<1x1x368x480xf16>

  // CHECK:       %0 = IE.Reshape(%arg0) {shape_value = [1, 12, 16, 11040]} : tensor<1x12x368x480xf16> -> tensor<1x12x16x11040xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       %1 = IE.Transpose(%0) {order_value = #NHCW} : tensor<1x12x16x11040xf16> -> tensor<1x16x12x11040xf16>
  // CHECK:       %2 = IE.AvgPool(%1) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x12x11040xf16> -> tensor<1x16x1x11040xf16>
  // CHECK-DAG:       %cst_0 = const.Declare tensor<1xf16> = dense<1.200000e+01> : tensor<1xf16>
  // CHECK:       %3 = IE.Multiply(%2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x11040xf16>, tensor<1xf16> -> tensor<1x16x1x11040xf16>
  // CHECK:       %4 = IE.Reshape(%3) {shape_value = [1, 1, 368, 480]} : tensor<1x16x1x11040xf16> -> tensor<1x1x368x480xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingAvoidingExpand2
func.func @ConvertReduceSumToPoolingAvoidingExpand2(%arg0: tensor<1x12x44x44xf16>) -> tensor<1x1x44x44xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x12x44x44xf16>, tensor<1xsi32> -> tensor<1x1x44x44xf16>
  return %1 : tensor<1x1x44x44xf16>

  // CHECK:       [[RESHAPE0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 12, 16, 121]} : tensor<1x12x44x44xf16> -> tensor<1x12x16x121xf16>
  // CHECK-NOT:   ReduceMax
  // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #NHCW} : tensor<1x12x16x121xf16> -> tensor<1x16x12x121xf16>
  // CHECK:       [[MAXPOOL0:%.*]] = IE.MaxPool([[TRANSPOSE0]]) {kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x12x121xf16> -> tensor<1x16x1x121xf16>
  // CHECK:       [[RESHAPE1:%.*]] = IE.Reshape([[MAXPOOL0]]) {shape_value = [1, 1, 44, 44]} : tensor<1x16x1x121xf16> -> tensor<1x1x44x44xf16>
  // CHECK:       return [[RESHAPE1]] : tensor<1x1x44x44xf16>
}

// CHECK-LABEL: @ConvertReduceSumToPoolingNegativeAxis
func.func @ConvertReduceSumToPoolingNegativeAxis(%arg0: tensor<1x10x10x40x40xf16>) -> tensor<1x10x10xf16> {
  %cst = const.Declare tensor<2xsi32> = dense<[-1, -2]> : tensor<2xsi64>, [#const.ConvertElemType<si32>]
  %1 = IE.ReduceSum(%arg0, %cst) : tensor<1x10x10x40x40xf16>, tensor<2xsi32> -> tensor<1x10x10xf16>
  return %1 : tensor<1x10x10xf16>

  // CHECK:       [[RESHAPE_0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 100, 1600, 1]} : tensor<1x10x10x40x40xf16> -> tensor<1x100x1600x1xf16>
  // CHECK-NOT:   ReduceSum
  // CHECK:       [[AVGPOOL_0:%.*]] = IE.AvgPool([[RESHAPE_0]]) {exclude_pads, kernel_size = [1600, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x100x1600x1xf16> -> tensor<1x100x1x1xf16>
  // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1xf16> = dense<1.600000e+03> : tensor<1xf16>
  // CHECK:       [[MULTIPLY_0:%.*]] = IE.Multiply([[AVGPOOL_0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x100x1x1xf16>, tensor<1xf16> -> tensor<1x100x1x1xf16>
  // CHECK:       [[RESHAPE_1:%.*]] = IE.Reshape([[MULTIPLY_0]]) {shape_value = [1, 10, 10]} : tensor<1x100x1x1xf16> -> tensor<1x10x10xf16>
}
