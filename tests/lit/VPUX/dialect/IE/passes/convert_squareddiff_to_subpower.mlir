//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-squareddiff-to-subpower %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertSquaredDiffToSubPower
func @ConvertSquaredDiffToSubPower(%arg0: tensor<1x64x128x64xf16>, %arg1: tensor<1x1x1x64xf16>) -> tensor<1x64x128x64xf16> {
    %0 = IE.SquaredDiff(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    return %0 : tensor<1x64x128x64xf16>


    // CHECK-NOT: IE.SquaredDiff
    // CHECK: [[SUBTRACT:%.*]] = IE.Subtract(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x64x128x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x64x128x64xf16>
    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[POWER:%.*]] = IE.Power([[SUBTRACT]], [[CST]]) {auto_broadcast = "NUMPY"} : tensor<1x64x128x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x128x64xf16>
    // CHECK: return [[POWER]] : tensor<1x64x128x64xf16>
}
