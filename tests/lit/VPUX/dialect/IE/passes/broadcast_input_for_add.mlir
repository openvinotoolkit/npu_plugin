//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --broadcast-input-for-add  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// -----

// CHECK-LABEL: @BroadcastAddInputForAdd
func.func @BroadcastAddInputForAdd(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x16x1xf16>) -> tensor<1x16x16x32xf16> {
    %ADD = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x16x32xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>

    return %ADD : tensor<1x16x16x32xf16>

    // CHECK-DAG:       %[[TARGET_SHAPE:.*]] = const.Declare tensor<4xsi64> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       %[[BROADCAST:.*]] = IE.Broadcast(%arg1, %[[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x16x16x1xf16>, tensor<4xsi64> -> tensor<1x16x16x32xf16>
    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %[[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return %[[ADD_RES]]
}

// -----

// CHECK-LABEL: @NotBroadcastAddSameInputShape
func.func @NotBroadcastAddSameInputShape(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
    %ADD = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    return %ADD : tensor<1x16x16x32xf16>

    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>
    // CHECK:       return %[[ADD_RES]]
}

// CHECK-LABEL: @BroadcastSingleNonTrivialDimAddInputForAdd
func.func @BroadcastSingleNonTrivialDimAddInputForAdd(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x16x32xf16> {
    %ADD = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

    return %ADD : tensor<1x16x16x32xf16>

    // CHECK-DAG:       %[[TARGET_SHAPE:.*]] = const.Declare tensor<4xsi64> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       %[[BROADCAST:.*]] = IE.Broadcast(%arg1, %[[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<4xsi64> -> tensor<1x16x16x32xf16>
    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %[[BROADCAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return %[[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastAddConstInput
func.func @BroadcastAddConstInput(%arg0: tensor<1x3x16xf16>) -> tensor<1x3x16xf16> {
    %CST = const.Declare tensor<1x1x16xf16> = dense<1.0> : tensor<1x1x16xf16>, [#const.ConvertElemType<f16>]
    %ADD = IE.Add(%arg0, %CST) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16xf16>, tensor<1x1x16xf16> -> tensor<1x3x16xf16>

    return %ADD : tensor<1x3x16xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x3x16xf16> = dense<1.000000e+00> : tensor<1x1x16xf16>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 16]>, #const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[ADD_RES:.*]] =  IE.Add(%arg0, %[[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16xf16>, tensor<1x3x16xf16> -> tensor<1x3x16xf16>
    // CHECK:       return %[[ADD_RES]]
}

// -----

// CHECK-LABEL: @NotBroadcastAddConstInput
func.func @NotBroadcastAddConstInput(%arg0: tensor<1x16x16xf16>) -> tensor<1x16x16xf16> {
    %CST = const.Declare tensor<1x1x16xf16> = dense<1.0> : tensor<1x1x16xf16>, [#const.ConvertElemType<f16>]
    %ADD = IE.Add(%arg0, %CST) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x16x16xf16>

    return %ADD : tensor<1x16x16xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x1x16xf16> = dense<1.000000e+00> : tensor<1x1x16xf16>, [#const.ConvertElemType<f16>]
    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %[[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x16x16xf16>
    // CHECK:       return %[[ADD_RES]]
}

// -----

// CHECK-LABEL: @BroadcastFQAddConstInput
func.func @BroadcastFQAddConstInput(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %CST = const.Declare tensor<1x1x30x30xf16> = dense<1.0> : tensor<1x1x30x30xf16>, [#const.ConvertElemType<f16>]
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq_1 = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %add = IE.Add(%fq_1, %CST)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16>, tensor<1x1x30x30xf16> -> tensor<1x3x30x30xf16>

    return %add : tensor<1x3x30x30xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1x3x30x30xf16> = dense<1.000000e+00> : tensor<1x1x30x30xf16>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 30, 30]>, #const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[RELU:%.*]] = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[RELU]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[ADD:%.*]] = IE.Add([[FQ_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[ADD]] : tensor<1x3x30x30xf16>
}
