//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-power-to-mult %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertPowerWithExponent2ToMult
func @ConvertPowerWithExponent2ToMult(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %cst_exponent = const.Declare tensor<1x1xf16> = dense<2.0> : tensor<1x1xf16>
    %power = IE.Power(%arg0, %cst_exponent) {auto_broadcast = "NUMPY"} : tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    return %power : tensor<1x16xf16>

    // CHECK-NOT:   IE.Power
    // CHECK:       %[[VAL0:.*]] = IE.Multiply(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    // CHECK:       return %[[VAL0]]
}
