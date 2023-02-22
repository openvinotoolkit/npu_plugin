//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --lower-VPU-to-VPUIP %s | FileCheck %s

//
// The 'lower-VPU-to-VPUIP' pipeline:
//
//   * Fully replaces VPU Dialect with VPUIP Dielect
//   * Changes all Value types from `tensor` to `memref`
//   * Adds result arguments to Function signature
//   * Inserts `VPUIP.Copy` to store result in output buffer
//   * Uses UPA SHAVE kernels for software ops
//

// CHECK: func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x1000xf16>
    // CHECK:       [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1000xf16>)

    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)

    // CHECK:       return [[VAR2]] : memref<1x1000xf16>
}

// -----

// CHECK: func @ReshapeInGraph([[ARG0:%.*]]: memref<1x512x1x1xf16>, [[ARG1:%.*]]: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
func @ReshapeInGraph(%arg0 : tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x512xf16>
    %1 = VPU.SoftMax(%0) {axisInd = 1} : tensor<1x512xf16> -> tensor<1x512xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf16> -> tensor<1x512x1x1xf16>
    return %2 : tensor<1x512x1x1xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x512x1x1xf16>) -> memref<1x512xf16>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x512xf16>
    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512xf16>)
    // CHECK:       [[VAR3:%.*]] = VPUIP.GenericReshape inputs([[VAR2]] : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       [[VAR4:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x512x1x1xf16>) outputs([[ARG1]] : memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       return [[VAR4]] : memref<1x512x1x1xf16>
}
