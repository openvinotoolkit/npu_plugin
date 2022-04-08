//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Fold
func @Fold(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    %0 = IE.Copy(%arg0) : tensor<1x3x16x16xf32> -> tensor<1x3x16x16xf32>
    return %0 : tensor<1x3x16x16xf32>

    // CHECK:       return %arg0 : tensor<1x3x16x16xf32>
}

// -----

// CHECK-LABEL: @FuseCopies
func @FuseCopies(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x3x16x16xf32> -> tensor<1x3x16x16xf32, {mem_space = @CMX_NN}>
    %1 = IE.Copy(%0) : tensor<1x3x16x16xf32, {mem_space = @CMX_NN}> -> tensor<1x3x16x16xf32>
    return %1 : tensor<1x3x16x16xf32>

    // CHECK:       return %arg0 : tensor<1x3x16x16xf32>
}
