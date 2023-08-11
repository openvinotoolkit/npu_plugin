//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-func-args-to-declarations --canonicalize --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func.func @WithoutInputs(%arg1: memref<10xf16, @DDR>) -> memref<10xf16, @DDR> {
    %cst = const.Declare memref<10xf16, @DDR> = dense<1.0> : tensor<10xf16>
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %2 = VPUIP.NNDMA inputs(%cst : memref<10xf16, @DDR>) outputs(%arg1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
    }
    return %arg1 : memref<10xf16, @DDR>

    //CHECK-DAG:    [[CST:%.+]] = const.Declare memref<10xf16, @DDR> = dense<1.000000e+00> : tensor<10xf16>
    //CHECK-DAG:    [[OUT0:%.+]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<10xf16, @DDR>

    //CHECK:        VPURT.Task attributes {isTrailingSWLayer = false}  {
    //CHECK:          %1 = VPUIP.NNDMA inputs([[CST]] : memref<10xf16, @DDR>) outputs([[OUT0]] : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
    //CHECK:        }

    //CHECK:        return %arg0 : memref<10xf16, @DDR>
}

// -----

func.func @SimpleGraph(%arg0: memref<10xf16, @DDR>, %arg1: memref<10xf16, @DDR>) -> memref<10xf16, @DDR> {
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %2 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16, @DDR>) outputs(%arg1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
    }
    return %arg1 : memref<10xf16, @DDR>

    //CHECK-DAG:    [[IN0:%.+]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<10xf16, @DDR>
    //CHECK-DAG:    [[OUT0:%.+]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<10xf16, @DDR>

    //CHECK:        VPURT.Task attributes {isTrailingSWLayer = false}  {
    //CHECK:          %2 = VPUIP.NNDMA inputs([[IN0]] : memref<10xf16, @DDR>) outputs([[OUT0]] : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
    //CHECK:        }

    //CHECK:        return %arg1 : memref<10xf16, @DDR>
}

// -----

func.func @TwoInOuts(%arg0: memref<2xf16, @DDR>, %arg1: memref<2xf16, @DDR>,
                %arg2: memref<2xf16, @DDR>, %arg3: memref<2xf16, @DDR>) -> (memref<2xf16, @DDR>, memref<2xf16, @DDR>) {
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %1 = VPUIP.NNDMA inputs(%arg0 : memref<2xf16, @DDR>) outputs(%arg2 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
    }
    VPURT.Task  attributes {isTrailingSWLayer = false}  {
      %1 = VPUIP.NNDMA inputs(%arg1 : memref<2xf16, @DDR>) outputs(%arg3 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
    }
    return %arg1, %arg2 : memref<2xf16, @DDR>, memref<2xf16, @DDR>

    //CHECK-DAG:    [[IN0:%.+]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<2xf16, @DDR>
    //CHECK-DAG:    [[IN1:%.+]] = VPURT.DeclareBuffer "NetworkInput" [1] <0> -> memref<2xf16, @DDR>
    //CHECK-DAG:    [[OUT0:%.+]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<2xf16, @DDR>
    //CHECK-DAG:    [[OUT1:%.+]] = VPURT.DeclareBuffer "NetworkOutput" [1] <0> -> memref<2xf16, @DDR>

    //CHECK:        VPURT.Task attributes {isTrailingSWLayer = false}  {
    //CHECK:          %4 = VPUIP.NNDMA inputs([[IN0]] : memref<2xf16, @DDR>) outputs([[OUT0]] : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
    //CHECK:        }
    //CHECK:        VPURT.Task attributes {isTrailingSWLayer = false}  {
    //CHECK:          %4 = VPUIP.NNDMA inputs([[IN1]] : memref<2xf16, @DDR>) outputs([[OUT1]] : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
    //CHECK:        }

    //CHECK:        return %arg1, %arg2 : memref<2xf16, @DDR>, memref<2xf16, @DDR>
}

// -----

func.func @WithReshapeNoChanges(%arg0: memref<1x512xf16, @DDR>, %arg1: memref<1x512xf16, @DDR>) -> memref<1x512xf16, @DDR> {
    %1 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x512x1x1xf16, @DDR>
    %2 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x512x1x1xf16, @DDR>
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.SoftMaxUPA {axisInd = 1 : i64} inputs(%1 : memref<1x512x1x1xf16, @DDR>)
            outputs(%2 : memref<1x512x1x1xf16, @DDR>) -> memref<1x512x1x1xf16, @DDR>
    }
    return %arg1 : memref<1x512xf16, @DDR>

    //CHECK:    %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x512x1x1xf16, @DDR>
    //CHECK:    %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x512x1x1xf16, @DDR>

    //CHECK:    VPURT.Task attributes
    //CHECK:    VPUIP.SoftMaxUPA

    //CHECK:    return %arg1 : memref<1x512xf16, @DDR>
}
