//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --broadcast-input-for-add  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// -----

// CHECK-LABEL: @BroadcastAddInputForAdd
func @BroadcastAddInputForAdd(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x16x1xf16>) -> tensor<1x16x16x32xf16> {
    %ADD = IE.Add(%arg0, %arg1) { auto_broadcast = "NUMPY" } : tensor<1x16x16x32xf16>, tensor<1x16x16x1xf16> -> tensor<1x16x16x32xf16>

    return %ADD : tensor<1x16x16x32xf16>

    // CHECK-DAG:       %[[TARGET_SHAPE:.*]] = const.Declare tensor<4xsi64> = dense<[1, 16, 16, 32]> : tensor<4xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       %[[BROADCAST:.*]] = IE.Broadcast(%arg1, %[[TARGET_SHAPE]])
    // CHECK-SAME:      {mode = "NUMPY"} : tensor<1x16x16x1xf16>, tensor<4xsi64> -> tensor<1x16x16x32xf16>
    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %[[BROADCAST]]) {auto_broadcast = "NUMPY"} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    // CHECK:       return %[[ADD_RES]]
}

// -----

// CHECK-LABEL: @NotBroadcastAddSameInputShape
func @NotBroadcastAddSameInputShape(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
    %ADD = IE.Add(%arg0, %arg1) { auto_broadcast = "NUMPY" } : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>

    return %ADD : tensor<1x16x16x32xf16>

    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x16x16x32xf16>, tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16>
    // CHECK:       return %[[ADD_RES]]
}

// -----

// CHECK-LABEL: @NotBroadcastAddInputForScaleShift
func @NotBroadcastAddInputForScaleShift(%arg0: tensor<1x16x16xf16>, %arg1: tensor<1x1x16xf16>) -> tensor<1x16x16xf16> {
    %ADD = IE.Add(%arg0, %arg1) { auto_broadcast = "NUMPY" } : tensor<1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x16x16xf16>

    return %ADD : tensor<1x16x16xf16>

    // CHECK:       %[[ADD_RES:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x16x16xf16>
    // CHECK:       return %[[ADD_RES]]
}
