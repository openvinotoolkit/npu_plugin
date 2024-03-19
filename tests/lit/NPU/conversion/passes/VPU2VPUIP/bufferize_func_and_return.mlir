//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --bufferize-func-and-return %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK: func.func @SingleInput({{[^:]+}}: memref<1x1x1x1000xf16>)
func.func @SingleInput(%input: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3, padSize = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %output: tensor<1x1x1x1000xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x1x1x1000xf16>
}

// -----

// CHECK: func.func @OnlyOneOutput() -> memref<1x2x2x2xf16> {
func.func @OnlyOneOutput() -> tensor<1x2x2x2xf16> {
    %output = const.Declare tensor<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.ConvertElemType<f16>]
    return %output : tensor<1x2x2x2xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x2x2x2xf16>
}

// -----

// CHECK: func.func @TwoInputs({{[^:]+}}: memref<1x16x16x16xf16>, {{[^:]+}}: memref<1x16x16x16xf16>) -> memref<1x32x16x16xf16> {
func.func @TwoInputs(%input0: tensor<1x16x16x16xf16>, %input1: tensor<1x16x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %output = VPU.Concat(%input0, %input1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x32x16x16xf16>
    return %output : tensor<1x32x16x16xf16>

    // CHECK: return
    // CHECK-SAME: memref<1x32x16x16xf16>
}

// -----

module @Convolution {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG:[^:]+]]: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
    func.func @foo1(%arg0: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
        %0 = VPU.Slice %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
        %1 = VPU.Slice %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        return %0, %1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

        // CHECK: return
        // CHECK-SAME: memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG:[^:]+]]: memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
    func.func @foo2(%input: tensor<1x4x60x60xf16>) -> tensor<1x4x60x60xf16> {
        return %input: tensor<1x4x60x60xf16>
        // CHECK: return [[ARG]] : memref<1x4x60x60xf16>
    }

    // CHECK: func.func @main([[ARG:[^:]+]]: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
    func.func @main(%arg0: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
        %0:2 = call @foo1(%arg0) : (tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>)
        %1 = call @foo2(%0#0) : (tensor<1x4x60x60xf16>) -> tensor<1x4x60x60xf16>
        return %1, %0#1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

        // CHECK: [[FOO1_RES:%.+]]:2 = call @foo1([[ARG]]) : (memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0) : (memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: return [[FOO2_RES]], [[FOO1_RES]]#1 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }
}
