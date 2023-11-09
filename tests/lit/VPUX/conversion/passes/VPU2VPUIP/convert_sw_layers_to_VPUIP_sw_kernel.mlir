//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-sw-layers-to-VPUIP-sw-kernel %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func.func @SingleSWLayer(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3, padSize = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %0: tensor<1x1x1x1000xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x1x1x1000xf16> to memref<1x1x1x1000xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1x1x1000xf16>) outputs([[VAR0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[VAR1]] as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
// CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 3]}(%arg1, %arg2) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x1x1x1000xf16>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x1x1x1000xf16> to tensor<1x1x1x1000xf16>
// CHECK: return [[VAR6]] : tensor<1x1x1x1000xf16>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func.func @MemPermuteSWLayer(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16, {order = #NHWC}> {
    %0 = VPU.MemPermute(%arg0) {mem_perm = #NHWC, dst_order = #NHWC} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC}>
    return %0: tensor<1x2x3x4xf16, {order = #NHWC}>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf16> to memref<1x2x3x4xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x2x3x4xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x2x3x4xf16>) outputs([[VAR0]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs([[VAR1]] as %arg1: memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>{
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   [2, 0, 1, 3]
// CHECK:   ]}(%arg1, %arg2) : memref<1x2x3x4xf16, [@CMX_NN, 0]>, memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x2x3x4xf16, #NHWC>) -> memref<1x2x3x4xf16, #NHWC>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast %5 : memref<1x2x3x4xf16, #NHWC> to tensor<1x2x3x4xf16, {order = #NHWC}>
// CHECK: return [[VAR6]] : tensor<1x2x3x4xf16, {order = #NHWC}>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func.func @ReorderSWLayer(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16, {order = #NHWC}> {
    %0 = VPU.MemPermute(%arg0) {mem_perm = #NHWC, dst_order = #NHWC} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC}>
    return %0: tensor<1x2x3x4xf16, {order = #NHWC}>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf16> to memref<1x2x3x4xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x2x3x4xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x2x3x4xf16>) outputs([[VAR0]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs([[VAR1]] as %arg1: memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>{
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   [2, 0, 1, 3]
// CHECK:   ]}(%arg1, %arg2) : memref<1x2x3x4xf16, [@CMX_NN, 0]>, memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x2x3x4xf16, #NHWC>) -> memref<1x2x3x4xf16, #NHWC>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x2x3x4xf16, #NHWC> to tensor<1x2x3x4xf16, {order = #NHWC}>
// CHECK: return [[VAR6]] : tensor<1x2x3x4xf16, {order = #NHWC}>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {
// CHECK: module @VPU.SW  {
// CHECK-NEXT: func.func private @builtin_Sigmoid(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
// CHECK-NEXT: func.func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK-NEXT: func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK: func.func @ThreeSWLayers(%arg0: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x1000xf16> {
func.func @ThreeSWLayers(%arg0: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %1 = VPU.Sigmoid(%0) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %2 = VPU.Slice %1 [0, 0, 0, 1000] [1, 1, 1, 1000] : tensor<1x1x1x2000xf16> to tensor<1x1x1x1000xf16>
    %3 = VPU.SoftMax(%2) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    return %3 : tensor<1x1x1x1000xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x1x1x2000xf16> to memref<1x1x1x2000xf16>
// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1x1x2000xf16>) outputs([[VAR2]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[VAR3]] as %arg1: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR4]] as %arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
// CHECK:         VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg1, %arg2) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK:       }
// CHECK: [[VAR20:%.*]] = memref.alloc() : memref<1x1x1x2000xf16>
// CHECK: [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR5]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR20]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

// CHECK: [[VAR7:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR8:%.*]] = VPUIP.Copy inputs([[VAR6]] : memref<1x1x1x2000xf16>) outputs([[VAR7]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR9:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR10:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR8]] as %arg1: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR9]] as %arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
// CHECK:       VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK:  }
// CHECK: [[VAR21:%.*]] = memref.alloc() : memref<1x1x1x2000xf16>
// CHECK: [[VAR11:%.*]] = VPUIP.Copy inputs([[VAR10]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR21]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

// CHECK: [[VAR22:%.*]] = builtin.unrealized_conversion_cast [[VAR11]] : memref<1x1x1x2000xf16> to tensor<1x1x1x2000xf16>
// CHECK: [[VAR23:%.*]] = VPU.Slice [[VAR22]] [0, 0, 0, 1000] [1, 1, 1, 1000] : tensor<1x1x1x2000xf16> to tensor<1x1x1x1000xf16>
// CHECK: [[VAR24:%.*]] = builtin.unrealized_conversion_cast [[VAR23]] : tensor<1x1x1x1000xf16> to memref<1x1x1x1000xf16>

// CHECK: [[VAR13:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR14:%.*]] = VPUIP.Copy inputs([[VAR24]] : memref<1x1x1x1000xf16>) outputs([[VAR13]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR15:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR16:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[VAR14]] as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR15]] as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
// CHECK:  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg1, %arg2) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK:  }
// CHECK: [[VAR25:%.*]] = memref.alloc() : memref<1x1x1x1000xf16>
// CHECK: [[VAR17:%.*]] = VPUIP.Copy inputs([[VAR16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR25]] : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
// CHECK: [[VAR26:%.*]] = builtin.unrealized_conversion_cast [[VAR17]] : memref<1x1x1x1000xf16> to tensor<1x1x1x1000xf16>

// CHECK:  return [[VAR26]] : tensor<1x1x1x1000xf16>

}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {

// CHECK:  module @VPU.SW {
// CHECK-NEXT:      func.func private @builtin_ReduceMean(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_mean.cpp", VPU.kernel_entry = "reduce_mean"}
// CHECK-NEXT:      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

func.func @ReduceMean(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7xf16>) -> tensor<1x512x7xf16> {
    %0 = VPU.ReduceMean(%arg0) {axes_value = [2]} : tensor<1x512x7x7xf16> -> tensor<1x512x7xf16>
    return %0 : tensor<1x512x7xf16>

// CHECK: [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x512x7x7xf16> to memref<1x512x7x7xf16>
// CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x512x7x7xf16, [@CMX_NN, 0]>
// CHECK: [[VAR2:%.*]] = VPUIP.Copy inputs([[VAR0]] : memref<1x512x7x7xf16>) outputs([[VAR1]] : memref<1x512x7x7xf16, [@CMX_NN, 0]>) -> memref<1x512x7x7xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = memref.alloc() : memref<1x512x7xf16, [@CMX_NN, 0]>
// CHECK: [[RES:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_ReduceMean inputs([[VAR2]] as %arg2: memref<1x512x7x7xf16, [@CMX_NN, 0]>) outputs([[VAR3]] as %arg3: memref<1x512x7xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x7xf16, [@CMX_NN, 0]>{
// CHECK: VPUIP.SW.Kernel.run {attrs = [0, 1, [2]]}(%arg2, %arg3) : memref<1x512x7x7xf16, [@CMX_NN, 0]>, memref<1x512x7xf16, [@CMX_NN, 0]>
// CHECK: }
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x512x7xf16>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[RES]] : memref<1x512x7xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x512x7xf16>) -> memref<1x512x7xf16>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x512x7xf16> to tensor<1x512x7xf16>
// CHECK: return [[VAR6]] : tensor<1x512x7xf16>

}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {


func.func @ActivationLog(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x50x1x1xf16> to tensor<1x50x1x1xf16>
    %1 = VPU.Log(%0) : tensor<1x50x1x1xf16> -> tensor<1x50x1x1xf16>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<1x50x1x1xf16> to memref<1x50x1x1xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50x1x1xf16>) outputs(%arg1 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16>
    return %3 : memref<1x50x1x1xf16>

// CHECK: [[VAR0:%.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x50x1x1xf16> to tensor<1x50x1x1xf16>
// CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x50x1x1xf16, [@CMX_NN, 0]>
// CHECK: [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x50x1x1xf16>) outputs([[VAR1]] : memref<1x50x1x1xf16, [@CMX_NN, 0]>) -> memref<1x50x1x1xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = memref.alloc() : memref<1x50x1x1xf16, [@CMX_NN, 0]>
// CHECK: [[RES:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Log inputs([[VAR2]] as %arg2: memref<1x50x1x1xf16, [@CMX_NN, 0]>) outputs([[VAR3]] as %arg3: memref<1x50x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x50x1x1xf16, [@CMX_NN, 0]>{
// CHECK: VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x50x1x1xf16, [@CMX_NN, 0]>, memref<1x50x1x1xf16, [@CMX_NN, 0]>
// CHECK: }
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x50x1x1xf16>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[RES]] : memref<1x50x1x1xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x50x1x1xf16> to tensor<1x50x1x1xf16>
// CHECK: [[VAR7:%.*]] = builtin.unrealized_conversion_cast [[VAR6]] : tensor<1x50x1x1xf16> to memref<1x50x1x1xf16>
// CHECK: [[VAR8:%.*]] = VPUIP.Copy inputs([[VAR7]] : memref<1x50x1x1xf16>) outputs(%arg1 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16>
// CHECK: return [[VAR8]] : memref<1x50x1x1xf16>

}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {

  func.func @Convolution(%arg0: memref<1x16x64xf16>, %arg1: memref<1x4x21xf16>) -> memref<1x4x21xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x64xf16> to tensor<1x16x64xf16>
    %cst = const.Declare tensor<4x16x1x9xf16> = dense<2.0> : tensor<4x16x5xf16>, [#const.Reshape<[4, 16, 1, 5]>, #const.ExpandDilated<[1, 2]>]
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 16, 1, 64]} : tensor<1x16x64xf16> -> tensor<1x16x1x64xf16>
    %2 = VPU.Convolution(%1, %cst) {dilations = [1, 1], pads_begin = [0, 3], pads_end = [0, 2], strides = [1, 3]} : tensor<1x16x1x64xf16>, tensor<4x16x1x9xf16> -> tensor<1x4x1x21xf16>
    %3 = VPU.AffineReshape(%2) {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 4, 21]} : tensor<1x4x1x21xf16> -> tensor<1x4x21xf16>
    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x4x21xf16> to memref<1x4x21xf16>
    %5 = VPUIP.Copy inputs(%4 : memref<1x4x21xf16>) outputs(%arg1 : memref<1x4x21xf16>) -> memref<1x4x21xf16>
    return %5 : memref<1x4x21xf16>

// CHECK: [[VAR0:%.*]] =  builtin.unrealized_conversion_cast %arg0 : memref<1x16x64xf16> to tensor<1x16x64xf16>
// CHECK-DAG: [[CST:%.*]]  = const.Declare tensor<4x16x1x9xf16> = dense<2.000000e+00> : tensor<4x16x5xf16>, [#const.Reshape<[4, 16, 1, 5]>, #const.ExpandDilated<[1, 2]>]
// CHECK: [[VAR1:%.*]] = builtin.unrealized_conversion_cast [[CST]] : tensor<4x16x1x9xf16> to memref<4x16x1x9xf16>
// CHECK: [[VAR2:%.*]] = VPU.AffineReshape([[VAR0]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 16, 1, 64]} : tensor<1x16x64xf16> -> tensor<1x16x1x64xf16>
// CHECK: [[VAR3:%.*]] = builtin.unrealized_conversion_cast [[VAR2]] : tensor<1x16x1x64xf16> to memref<1x16x1x64xf16>
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x16x1x64xf16, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x16x1x64xf16>) outputs([[VAR4]] : memref<1x16x1x64xf16, [@CMX_NN, 0]>) -> memref<1x16x1x64xf16, [@CMX_NN, 0]>
// CHECK: [[VAR6:%.*]] = memref.alloc() : memref<4x16x1x9xf16, [@CMX_NN, 0]>
// CHECK: [[VAR7:%.*]] = VPUIP.Copy inputs([[VAR1]] : memref<4x16x1x9xf16>) outputs([[VAR6]] : memref<4x16x1x9xf16, [@CMX_NN, 0]>) -> memref<4x16x1x9xf16, [@CMX_NN, 0]>
// CHECK: [[VAR8:%.*]] = memref.alloc() : memref<1x4x1x21xf16, [@CMX_NN, 0]>
// CHECK: [[RESULT:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Convolution inputs([[VAR5]] as %arg2: memref<1x16x1x64xf16, [@CMX_NN, 0]>, [[VAR7]] as %arg3: memref<4x16x1x9xf16, [@CMX_NN, 0]>) outputs([[VAR8]] as %arg4: memref<1x4x1x21xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x1x21xf16, [@CMX_NN, 0]>{
// CHECK: VPUIP.SW.Kernel.run
// CHECK-SAME{LITERAL}: {attrs = [[1, 3], [0, 3], [0, 2], [1, 1], 1]}(%arg2, %arg3, %arg4) : memref<1x16x1x64xf16, [@CMX_NN, 0]>, memref<4x16x1x9xf16, [@CMX_NN, 0]>, memref<1x4x1x21xf16, [@CMX_NN, 0]>
// CHECK: }
// CHECK: [[VAR9:%.*]] = memref.alloc() : memref<1x4x1x21xf16>
// CHECK: [[VAR10:%.*]] = VPUIP.Copy inputs([[RESULT]] : memref<1x4x1x21xf16, [@CMX_NN, 0]>) outputs([[VAR9]] : memref<1x4x1x21xf16>) -> memref<1x4x1x21xf16>
// CHECK: [[VAR11:%.*]] = builtin.unrealized_conversion_cast [[VAR10]] : memref<1x4x1x21xf16> to tensor<1x4x1x21xf16>
// CHECK: [[VAR12:%.*]] = VPU.AffineReshape([[VAR11]])
// CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 4, 21]} : tensor<1x4x1x21xf16> -> tensor<1x4x21xf16>
// CHECK: [[VAR13:%.*]] = builtin.unrealized_conversion_cast [[VAR12]] : tensor<1x4x21xf16> to memref<1x4x21xf16>
// CHECK: [[VAR14:%.*]] = VPUIP.Copy inputs([[VAR13]] : memref<1x4x21xf16>) outputs(%arg1 : memref<1x4x21xf16>) -> memref<1x4x21xf16>
// CHECK: return [[VAR14]] : memref<1x4x21xf16>

}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_Interpolate(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, none, none, none, none, f64, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func.func @InterpolateSWLayerWithUnnecessaryScalingAxes(%arg0: tensor<1x128x1x1xf16>) -> tensor<1x128x32x32xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 1, 1], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 32, 32], initial_output_offset_attr = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 1.000000e+00, 3.200000e+00, 3.200000e+00], sizes_attr = [1, 128, 32, 32], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x1x1xf16> -> tensor<1x128x32x32xf16>

    return %0 : tensor<1x128x32x32xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x128x1x1xf16> to memref<1x128x1x1xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x128x1x1xf16>) outputs([[VAR0]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x128x32x32xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs([[VAR1]] as %arg1: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x128x32x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x32x32xf16, [@CMX_NN, 0]>{
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1, 1, 128, 1], [32, 32, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]
// CHECK:   ]}(%arg1, %arg2) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x32x32xf16, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x128x32x32xf16>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x128x32x32xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x128x32x32xf16>) -> memref<1x128x32x32xf16>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast %5 : memref<1x128x32x32xf16> to tensor<1x128x32x32xf16>
// CHECK: return [[VAR6]] : tensor<1x128x32x32xf16>
}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {

func.func @GroupConvolution(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x8x28x28xf16> {
  %cst = const.Declare tensor<8x8x3x3xf16> = dense<2.0> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
  %0 = VPU.GroupConvolution(%arg0, %cst) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x30x30xf16>, tensor<8x8x3x3xf16> -> tensor<1x8x28x28xf16>
  return %0 : tensor<1x8x28x28xf16>

// CHECK: [[VAR0:%.*]]  = builtin.unrealized_conversion_cast %arg0 : tensor<1x16x30x30xf16> to memref<1x16x30x30xf16>
// CHECK-DAG: [[CST:%.*]]   = const.Declare tensor<8x8x3x3xf16> = dense<2.000000e+00> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
// CHECK: [[VAR1:%.*]]  = builtin.unrealized_conversion_cast [[CST]] : tensor<8x8x3x3xf16> to memref<8x8x3x3xf16>
// CHECK: [[VAR2:%.*]]  = memref.alloc() : memref<1x16x30x30xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]]  = VPUIP.Copy inputs([[VAR0]] : memref<1x16x30x30xf16>) outputs([[VAR2]] : memref<1x16x30x30xf16, [@CMX_NN, 0]>) -> memref<1x16x30x30xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]]  = memref.alloc() : memref<8x8x3x3xf16, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]]  = VPUIP.Copy inputs([[VAR1]] : memref<8x8x3x3xf16>) outputs([[VAR4]] : memref<8x8x3x3xf16, [@CMX_NN, 0]>) -> memref<8x8x3x3xf16, [@CMX_NN, 0]>
// CHECK: [[VAR6:%.*]]  = memref.alloc() : memref<1x8x28x28xf16, [@CMX_NN, 0]>
// CHECK: [[VAR7:%.*]]  = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_GroupConvolution
// CHECK-SAME:          inputs([[VAR3]] as %arg1: memref<1x16x30x30xf16, [@CMX_NN, 0]>, [[VAR5]] as %arg2: memref<8x8x3x3xf16, [@CMX_NN, 0]>)
// CHECK-SAME:          outputs([[VAR6]] as %arg3: memref<1x8x28x28xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x28x28xf16, [@CMX_NN, 0]>{
// CHECK: [[VAR8:%.*]]  = memref.alloc() : memref<1x8x28x28xf16>
// CHECK: [[VAR9:%.*]]  = VPUIP.Copy inputs([[VAR7]] : memref<1x8x28x28xf16, [@CMX_NN, 0]>) outputs([[VAR8]] : memref<1x8x28x28xf16>) -> memref<1x8x28x28xf16>
// CHECK: [[VAR10:%.*]] = builtin.unrealized_conversion_cast [[VAR9]] : memref<1x8x28x28xf16> to tensor<1x8x28x28xf16>
// CHECK: return [[VAR10]] : tensor<1x8x28x28xf16>
}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

func.func @GroupConvolution(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x8x28x28xf16> {
  %cst = const.Declare tensor<8x8x3x3xf16> = dense<2.0> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
  %0 = VPU.GroupConvolution(%arg0, %cst) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x30x30xf16>, tensor<8x8x3x3xf16> -> tensor<1x8x28x28xf16>
  return %0 : tensor<1x8x28x28xf16>

// CHECK: [[VAR0:%.*]]  = builtin.unrealized_conversion_cast %arg0 : tensor<1x16x30x30xf16> to memref<1x16x30x30xf16>
// CHECK-DAG: [[CST:%.*]]   = const.Declare tensor<8x8x3x3xf16> = dense<2.000000e+00> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
// CHECK: [[VAR1:%.*]]  = builtin.unrealized_conversion_cast [[CST]] : tensor<8x8x3x3xf16> to memref<8x8x3x3xf16>
// CHECK: [[VAR2:%.*]]  = memref.alloc() : memref<1x16x30x30xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]]  = VPUIP.Copy inputs([[VAR0]] : memref<1x16x30x30xf16>) outputs([[VAR2]] : memref<1x16x30x30xf16, [@CMX_NN, 0]>) -> memref<1x16x30x30xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]]  = memref.alloc() : memref<8x8x3x3xf16, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]]  = VPUIP.Copy inputs([[VAR1]] : memref<8x8x3x3xf16>) outputs([[VAR4]] : memref<8x8x3x3xf16, [@CMX_NN, 0]>) -> memref<8x8x3x3xf16, [@CMX_NN, 0]>
// CHECK: [[VAR6:%.*]]  = memref.alloc() : memref<1x8x28x28xf16, [@CMX_NN, 0]>
// CHECK: [[VAR7:%.*]]  = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_GroupConvolution
// CHECK-SAME:          inputs([[VAR3]] as %arg1: memref<1x16x30x30xf16, [@CMX_NN, 0]>, [[VAR5]] as %arg2: memref<8x8x3x3xf16, [@CMX_NN, 0]>)
// CHECK-SAME:          outputs([[VAR6]] as %arg3: memref<1x8x28x28xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x28x28xf16, [@CMX_NN, 0]>{
// CHECK: [[VAR8:%.*]]  = memref.alloc() : memref<1x8x28x28xf16>
// CHECK: [[VAR9:%.*]]  = VPUIP.Copy inputs([[VAR7]] : memref<1x8x28x28xf16, [@CMX_NN, 0]>) outputs([[VAR8]] : memref<1x8x28x28xf16>) -> memref<1x8x28x28xf16>
// CHECK: [[VAR10:%.*]] = builtin.unrealized_conversion_cast [[VAR9]] : memref<1x8x28x28xf16> to tensor<1x8x28x28xf16>
// CHECK: return [[VAR10]] : tensor<1x8x28x28xf16>
}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {
// CHECK:  module @VPU.SW {
// CHECK-NEXT:      func.func private @builtin_StridedSlice(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none, none, none) attributes {VPU.kernel_code = "single_shave_stridedslice.cpp", VPU.kernel_entry = "single_shave_stridedslice"}
// CHECK-NEXT:      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }
func.func @StridedSlice2Dim(%arg0: tensor<3x40x40x15xf16>) -> tensor<3x40x20x5xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x20x5xf16>
    return %0 : tensor<3x40x20x5xf16>

    // CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x40x40x15xf16> to memref<3x40x40x15xf16>
    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<3x40x40x15xf16>) outputs([[VAR0]] : memref<3x40x40x15xf16, [@CMX_NN, 0]>) -> memref<3x40x40x15xf16, [@CMX_NN, 0]>

    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<3x40x20x5xf16, [@CMX_NN, 0]>
    // CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_StridedSlice inputs([[VAR1]] as %arg1: memref<3x40x40x15xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<3x40x20x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<3x40x20x5xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [
    // CHECK:   [0, 0, 0, 0], [3, 40, 40, 15], [1, 1, 2, 3]
    // CHECK:   ]}(%arg1, %arg2) : memref<3x40x40x15xf16, [@CMX_NN, 0]>, memref<3x40x20x5xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[VAR4:%.*]] = memref.alloc() : memref<3x40x20x5xf16>
    // CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<3x40x20x5xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<3x40x20x5xf16>) -> memref<3x40x20x5xf16>
    // CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<3x40x20x5xf16> to tensor<3x40x20x5xf16>
    // CHECK: return [[VAR6]] : tensor<3x40x20x5xf16>
}

}

// -----

module @Test attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

func.func @StridedSlice1Dim(%arg0: tensor<3x40x40x15xf16>) -> tensor<3x40x40x5xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    return %0 : tensor<3x40x40x5xf16>

    // CHECK: [[VAR0:%.*]] = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    // CHECK: return [[VAR0]] : tensor<3x40x40x5xf16>

}

}
