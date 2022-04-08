//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-sw-layers-to-Affine %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SingleSWLayer(%arg0: memref<1000xf16>, %arg1: memref<1000xf16>) -> memref<1000xf16> {
    %0 = IERT.Cos inputs(%arg0 : memref<1000xf16>) outputs(%arg1 : memref<1000xf16>) -> memref<1000xf16>
    return %0: memref<1000xf16>
}

// CHECK: func @SingleSWLayer(%[[NUMA:.+]]: memref<1000xf16>, %[[NUMC:.+]]: memref<1000xf16>) -> memref<1000xf16> {
// CHECK: affine.for  %[[NUMB:.+]] = 0 to 1000 {
// CHECK: [[LD:%.+]] = affine.load %[[NUMA]][%[[NUMB]]] : memref<1000xf16>
// CHECK: [[RES:%.+]] = math.cos [[LD]] : f16

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SingleSWLayer(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) -> memref<1000xf32> {
    %0 = IERT.Cos inputs(%arg0 : memref<1000xf32>) outputs(%arg1 : memref<1000xf32>) -> memref<1000xf32>
    return %0: memref<1000xf32>
}

// CHECK: func @SingleSWLayer(%[[NUMA:.+]]: memref<1000xf32>, %[[NUMC:.+]]: memref<1000xf32>) -> memref<1000xf32> {
// CHECK: affine.for  %[[NUMB:.+]] = 0 to 1000 {
// CHECK: [[LD:%.+]] = affine.load %[[NUMA]][%[[NUMB]]] : memref<1000xf32>
// CHECK: [[RES:%.+]] = math.cos [[LD]] : f32

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SingleSWLayer(%arg0: memref<1000xf16>, %arg1: memref<1000xf16>) -> memref<1000xf16> {
    %0 = IERT.HSwish {axisInd = 3} inputs(%arg0 : memref<1000xf16>) outputs(%arg1 : memref<1000xf16>) -> memref<1000xf16>
    return %0: memref<1000xf16>
}

// CHECK: func @SingleSWLayer(%[[NUMA:.+]]: memref<1000xf16>, %[[NUMC:.+]]: memref<1000xf16>) -> memref<1000xf16> {
// CHECK: affine.for %[[NUMB:.+]] = 0 to 1000 {
// CHECK: %cst = arith.constant 3.000000e+00 : f16
// CHECK: %[[VAL2:.+]] = arith.addf %[[VAL1:.+]], %cst : f16
// CHECK: %[[VAL3:.+]] = arith.maxf %[[VAL2]], %cst_0 : f16
// CHECK: %[[VAL4:.+]] = arith.minf %[[VAL3]], %cst_1 : f16
// CHECK: affine.store %[[VAL6:.+]], %[[NUMC]][%[[NUMB]]] : memref<1000xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SingleSWLayer(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) -> memref<1000xf32> {
    %0 = IERT.HSwish {axisInd = 3} inputs(%arg0 : memref<1000xf32>) outputs(%arg1 : memref<1000xf32>) -> memref<1000xf32>
    return %0: memref<1000xf32>
}

// CHECK: func @SingleSWLayer(%[[NUMA:.+]]: memref<1000xf32>, %[[NUMC:.+]]: memref<1000xf32>) -> memref<1000xf32> {
// CHECK: affine.for %[[NUMB:.+]] = 0 to 1000 {
// CHECK: %cst = arith.constant 3.000000e+00 : f32
// CHECK: %[[VAL2:.+]] = arith.addf %[[VAL1:.+]], %cst : f32
// CHECK: %[[VAL3:.+]] = arith.maxf %[[VAL2]], %cst_0 : f32
// CHECK: %[[VAL4:.+]] = arith.minf %[[VAL3]], %cst_1 : f32
// CHECK: affine.store %[[VAL6:.+]], %[[NUMC]][%[[NUMB]]] : memref<1000xf32>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SingleSWLayer(%arg0: memref<1000xf16>, %arg1: memref<1000xf16>) -> memref<1000xf16> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1000xf16>) outputs(%arg1 : memref<1000xf16>) -> memref<1000xf16>
    return %0: memref<1000xf16>
}

// CHECK: func @SingleSWLayer(%[[NUMA:.+]]: memref<1000xf16>, %[[NUMC:.+]]: memref<1000xf16>) -> memref<1000xf16> {
// CHECK: %[[VAL1:.+]] = memref.alloc() : memref<1xf16>
// CHECK: %[[VAL0:.+]] = memref.alloc() : memref<1000xf16>
// CHECK: affine.for %[[NUMB:.+]] = 0 to 1000 {
// CHECK: %[[VAL5:.+]] = affine.load %[[VAL1]][%c0] : memref<1xf16>
// CHECK: affine.store %[[VAL6:.+]], %[[VAL1]][%c0] : memref<1xf16>
// CHECK: affine.for %[[NUMB:.+]] = 0 to 1000 {
// CHECK: %[[VAL5:.+]] = affine.load %[[VAL1:.+]][%c0] : memref<1xf16>
// CHECK: %[[VAL6:.+]] = arith.divf %[[VAL4:.+]], %[[VAL5]] : f16
// CHECK: affine.store %[[VAL6]], %[[NUMC]][%[[NUMB]]] : memref<1000xf16>
// CHECK: memref.dealloc %[[VAL1]] : memref<1xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SingleSWLayer(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) -> memref<1000xf32> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1000xf32>) outputs(%arg1 : memref<1000xf32>) -> memref<1000xf32>
    return %0: memref<1000xf32>
}

// CHECK: func @SingleSWLayer(%[[NUMA:.+]]: memref<1000xf32>, %[[NUMC:.+]]: memref<1000xf32>) -> memref<1000xf32> {
// CHECK: %[[VAL1:.+]] = memref.alloc() : memref<1xf32>
// CHECK: %[[VAL0:.+]] = memref.alloc() : memref<1000xf32>
// CHECK: affine.for %[[NUMB:.+]] = 0 to 1000 {
// CHECK: %[[VAL5:.+]] = affine.load %[[VAL1]][%c0] : memref<1xf32>
// CHECK: affine.store %[[VAL6:.+]], %[[VAL1]][%c0] : memref<1xf32>
// CHECK: affine.for %[[NUMB:.+]] = 0 to 1000 {
// CHECK: %[[VAL5:.+]] = affine.load %[[VAL1:.+]][%c0] : memref<1xf32>
// CHECK: %[[VAL6:.+]] = arith.divf %[[VAL4:.+]], %[[VAL5]] : f32
// CHECK: affine.store %[[VAL6]], %[[NUMC]][%[[NUMB]]] : memref<1000xf32>
// CHECK: memref.dealloc %[[VAL1]] : memref<1xf32>
