//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --convert-sw-layers-to-Affine --convert-Affine-to-LLVM %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1000xf16>, %arg1: memref<1000xf16>) -> memref<1000xf16> {
    %0 = IERT.Cos inputs(%arg0 : memref<1000xf16>) outputs(%arg1 : memref<1000xf16>) -> memref<1000xf16>
    return %0: memref<1000xf16>
}

}

// CHECK: llvm.func @SingleSWLayer(
// CHECK: %[[VAL33:.+]] = llvm.load %[[VAL32:.+]] : !llvm.ptr<f16>
// CHECK: %[[VAL34:.+]] = "llvm.intr.cos"(%[[VAL33]]) : (f16) -> f16
// CHECK: llvm.store %[[VAL34]], %[[VAL36:.+]] : !llvm.ptr<f16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) -> memref<1000xf32> {
    %0 = IERT.Cos inputs(%arg0 : memref<1000xf32>) outputs(%arg1 : memref<1000xf32>) -> memref<1000xf32>
    return %0: memref<1000xf32>
}

}

// CHECK: llvm.func @SingleSWLayer(
// CHECK: %[[VAL33:.+]] = llvm.load %[[VAL32:.+]] : !llvm.ptr<f32>
// CHECK: %[[VAL34:.+]] = "llvm.intr.cos"(%[[VAL33]]) : (f32) -> f32
// CHECK: llvm.store %[[VAL34]], %[[VAL36:.+]] : !llvm.ptr<f32>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1000xf16>, %arg1: memref<1000xf16>) -> memref<1000xf16> {
    %0 = IERT.HSwish {axisInd = 3} inputs(%arg0 : memref<1000xf16>) outputs(%arg1 : memref<1000xf16>) -> memref<1000xf16>
    return %0: memref<1000xf16>
}

}

// CHECK: llvm.func @SingleSWLayer(
// CHECK: %[[VAL34:.+]] = llvm.mlir.constant(3.000000e+00 : f16) : f16
// CHECK: %[[VAL35:.+]] = llvm.fadd %[[VAL33:.+]], %[[VAL34]]  : f16
// CHECK: %[[VAL46:.+]] = llvm.fdiv %[[VAL45:.+]], %[[VAL41:.+]]  : f16
// CHECK: %[[VAL47:.+]] = llvm.fmul %[[VAL46]], %[[VAL33]]  : f16
// CHECK: llvm.store %[[VAL47]], %[[VAL49:.+]] : !llvm.ptr<f16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) -> memref<1000xf32> {
    %0 = IERT.HSwish {axisInd = 3} inputs(%arg0 : memref<1000xf32>) outputs(%arg1 : memref<1000xf32>) -> memref<1000xf32>
    return %0: memref<1000xf32>
}

}

// CHECK: llvm.func @SingleSWLayer(
// CHECK: %[[VAL34:.+]] = llvm.mlir.constant(3.000000e+00 : f32) : f32
// CHECK: %[[VAL35:.+]] = llvm.fadd %[[VAL33:.+]], %[[VAL34]]  : f32
// CHECK: %[[VAL46:.+]] = llvm.fdiv %[[VAL45:.+]], %[[VAL41:.+]]  : f32
// CHECK: %[[VAL47:.+]] = llvm.fmul %[[VAL46]], %[[VAL33]]  : f32
// CHECK: llvm.store %[[VAL47]], %[[VAL49:.+]] : !llvm.ptr<f32>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1000xf16>, %arg1: memref<1000xf16>) -> memref<1000xf16> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1000xf16>) outputs(%arg1 : memref<1000xf16>) -> memref<1000xf16>
    return %0: memref<1000xf16>
}

}

// CHECK: %[[VAL61:.+]] = llvm.load %[[VAL60:.+]] : !llvm.ptr<f16>
// CHECK: %[[VAL62:.+]] = "llvm.intr.exp"(%[[VAL61]]) : (f16) -> f16
// CHECK: llvm.store %[[VAL67:.+]], %[[VAL69:.+]] : !llvm.ptr<f16>
// CHECK: %[[VAL78:.+]] = llvm.load %[[VAL77:.+]] : !llvm.ptr<f16>
// CHECK: %[[VAL79:.+]] = "llvm.intr.exp"(%[[VAL78]]) : (f16) -> f16
// CHECK: %[[VAL83:.+]] = llvm.load %[[VAL82:.+]] : !llvm.ptr<f16>
// CHECK: %[[VAL84:.+]] = llvm.fdiv %[[VAL79]], %[[VAL83]]  : f16
// CHECK: llvm.store %[[VAL84]], %[[VAL86:.+]] : !llvm.ptr<f16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) -> memref<1000xf32> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1000xf32>) outputs(%arg1 : memref<1000xf32>) -> memref<1000xf32>
    return %0: memref<1000xf32>
}

}

// CHECK: %[[VAL61:.+]] = llvm.load %[[VAL60:.+]] : !llvm.ptr<f32>
// CHECK: %[[VAL62:.+]] = "llvm.intr.exp"(%[[VAL61]]) : (f32) -> f32
// CHECK: llvm.store %[[VAL67:.+]], %[[VAL69:.+]] : !llvm.ptr<f32>
// CHECK: %[[VAL78:.+]] = llvm.load %[[VAL77:.+]] : !llvm.ptr<f32>
// CHECK: %[[VAL79:.+]] = "llvm.intr.exp"(%[[VAL78]]) : (f32) -> f32
// CHECK: %[[VAL83:.+]] = llvm.load %[[VAL82:.+]] : !llvm.ptr<f32>
// CHECK: %[[VAL84:.+]] = llvm.fdiv %[[VAL79]], %[[VAL83]]  : f32
// CHECK: llvm.store %[[VAL84]], %[[VAL86:.+]] : !llvm.ptr<f32>
