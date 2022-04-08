//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --lower-VPU-to-VPUIP %s | FileCheck %s

//
// The 'lower-VPU-to-VPUIP' pipeline:
//
//   * Fully replaces VPU Dialect with VPUIP Dielect
//   * Changes all Value types from `tensor` to `memref`
//   * Adds result arguments to Function signature
//   * Inserts `VPUIP.Copy` to store result in output buffer
//   * Uses activation SHAVE kernels `VPUIP.SW.Kernel` for software ops
//

// CHECK:       VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
// CHECK:       module @VPU.SW  {
// CHECK:           func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64)
// CHECK-SAME:           attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK:           func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK:       }

// CHECK: func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:                  outputs([[VAR0]] : memref<1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       [[VAR2:%.+]] = memref.alloc() : memref<1x1000xf16, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
    // CHECK-SAME:      @VPU.SW::@builtin_SoftMax inputs([[VAR1]] as %arg2: memref<1x1000xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:                                outputs([[VAR2]] as %arg3: memref<1x1000xf16, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:               -> memref<1x1000xf16, [@CMX_NN, 0]>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3)
    // CHECK-SAME:               : memref<1x1000xf16, [@CMX_NN, 0]>, memref<1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       [[VAR4:%.+]] = memref.alloc() : memref<1x1000xf16>
    // CHECK:       [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR3]] : memref<1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x1000xf16>) -> memref<1x1000xf16>
    // CHECK:       [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<1x1000xf16>) outputs([[ARG1]] : memref<1x1000xf16>) -> memref<1x1000xf16>
    // CHECK:       return [[VAR6]] : memref<1x1000xf16>
}

// -----

// CHECK:       VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
// CHECK:       module @VPU.SW  {
// CHECK:           func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64)
// CHECK-SAME:           attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK:           func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK:       }

// CHECK: func @ReshapeInGraph([[ARG0:%.*]]: memref<1x512x1x1xf16>, [[ARG1:%.*]]: memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16> {
func @ReshapeInGraph(%arg0 : tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16> {
    %0 = VPU.Reshape(%arg0) {shape_value = [1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x512xf16>
    %1 = VPU.SoftMax(%0) {axisInd = 1} : tensor<1x512xf16> -> tensor<1x512xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 512, 1, 1]} : tensor<1x512xf16> -> tensor<1x512x1x1xf16>
    return %2 : tensor<1x512x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.GenericReshape inputs(%arg0 : memref<1x512x1x1xf16>) -> memref<1x512xf16>
    // CHECK:       [[VAR1:%.+]] = memref.alloc() : memref<1x512xf16, [@CMX_NN, 0]>
    // CHECK:       [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR0]] : memref<1x512xf16>)
    // CHECK-SAME:                  outputs([[VAR1]] : memref<1x512xf16, [@CMX_NN, 0]>) -> memref<1x512xf16, [@CMX_NN, 0]>
    // CHECK:       [[VAR3:%.+]] = memref.alloc() : memref<1x512xf16, [@CMX_NN, 0]>

    // CHECK:       [[VAR4:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
    // CHECK-SAME:      @VPU.SW::@builtin_SoftMax inputs([[VAR2]] as %arg2: memref<1x512xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:                                outputs([[VAR3]] as %arg3: memref<1x512xf16, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:               -> memref<1x512xf16, [@CMX_NN, 0]>{
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3)
    // CHECK-SAME:               : memref<1x512xf16, [@CMX_NN, 0]>, memref<1x512xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       [[VAR5:%.+]] = memref.alloc() : memref<1x512xf16>
    // CHECK:       [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x512xf16, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x512xf16>) -> memref<1x512xf16>
    // CHECK:       [[VAR7:%.+]] = VPUIP.GenericReshape inputs([[VAR6]] : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR7]] : memref<1x512x1x1xf16>) outputs([[ARG1]] : memref<1x512x1x1xf16>) -> memref<1x512x1x1xf16>
    // CHECK:       return [[VAR8]] : memref<1x512x1x1xf16>
}
