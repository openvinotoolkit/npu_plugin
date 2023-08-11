//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-quant-cast-and-clamp  %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType0 = !quant.uniform<u8:f16, 0.0022939644607843138>
!qElemType1 = !quant.uniform<u8:f16, 0.0011469822303921569>

// CHECK: !qElemType0 = !quant.uniform<u8:f16, 0.0011469822303921569>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0022939644607843138>

// CHECK-LABEL: @SwapQuantizeCast
func.func @SwapQuantizeCast(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x8x8x!qElemType1> {
      %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType0>
      %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x16x8x8x!qElemType0> -> tensor<1x16x8x8x!qElemType1>
      %2 = IE.Clamp(%1) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x16x8x8x!qElemType1> -> tensor<1x16x8x8x!qElemType1>

      return %2 : tensor<1x16x8x8x!qElemType1>

      // CHECK: [[VAR0:%.+]] = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
      // CHECK: [[VAR1:%.+]] = IE.Clamp([[VAR0]]) {max = 1.000000e+01 : f64, min = 2.000000e+00 : f64}
      // CHECK: [[VAR2:%.+]] = IE.QuantizeCast([[VAR1]]) {dstElemType = !qElemType0}
      // CHECK: return [[VAR2]] : tensor<1x16x8x8x!qElemType0>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.0022939644607843138>
!qElemType1 = !quant.uniform<u8:f16, 0.0011469822303921569>

// CHECK: !qElemType0 = !quant.uniform<u8:f16, 0.0011469822303921569>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0022939644607843138>

// CHECK-LABEL: @NoSwapTwoConsumers
func.func @NoSwapTwoConsumers(%arg0: tensor<1x16x8x8xf16>) -> (tensor<1x16x8x8x!qElemType1>, tensor<1x16x8x8x!qElemType1>) {
      %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16> -> tensor<1x16x8x8x!qElemType0>
      %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x16x8x8x!qElemType0> -> tensor<1x16x8x8x!qElemType1>
      %2 = IE.Clamp(%1) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x16x8x8x!qElemType1> -> tensor<1x16x8x8x!qElemType1>

      return %1, %2 : tensor<1x16x8x8x!qElemType1>, tensor<1x16x8x8x!qElemType1>

      // CHECK: [[VAR0:%.+]] = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
      // CHECK: [[VAR1:%.+]] = IE.QuantizeCast([[VAR0]]) {dstElemType = !qElemType0}
      // CHECK: [[VAR2:%.+]] = IE.Clamp([[VAR1]]) {max = 5.000000e+00 : f64, min = 1.000000e+00 : f64}

      // CHECK: return [[VAR1]], [[VAR2]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.0022939644607843138>

// CHECK-LABEL: @NoSwapIntInput
func.func @NoSwapIntInput(%arg0: tensor<1x16x8x8xui8>) -> tensor<1x16x8x8x!qElemType0> {
      %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType0} : tensor<1x16x8x8xui8> -> tensor<1x16x8x8x!qElemType0>
      %1 = IE.Clamp(%0) {max = 5.000000e+00, min = 1.000000e+00} : tensor<1x16x8x8x!qElemType0> -> tensor<1x16x8x8x!qElemType0>

      return %1 : tensor<1x16x8x8x!qElemType0>

      // CHECK: [[VAR0:%.+]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType}
      // CHECK: [[VAR1:%.+]] = IE.Clamp([[VAR0]]) {max = 5.000000e+00 : f64, min = 1.000000e+00 : f64}

      // CHECK: return [[VAR1]]
}
