//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @BroadcastFoldFold
func @BroadcastFoldFold(%arg0 : tensor<1x8x4x4xf32>)-> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<1> : tensor<4xsi64>
    %1 = IE.Broadcast(%arg0, %0) {mode = "BIDIRECTIONAL"} : tensor<1x8x4x4xf32>, tensor<4xsi64> -> tensor<1x8x4x4xf32>
    return %1 : tensor<1x8x4x4xf32>

    // CHECK-NOT: IE.Broadcast
    // CHECK:     return %arg0
}

