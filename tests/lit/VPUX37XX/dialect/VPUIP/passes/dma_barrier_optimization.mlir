//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --dma-barrier-optimization %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!Input_DDR = type memref<1x3x224x224x!qElemType0, #NHWC, @DDR>
!Output_DDR = type memref<1x16x224x224x!qElemType0, #NHWC, @DDR>

//CHECK-LABEL: @DMABarrierOptimization
func @DMABarrierOptimization() -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %input0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [150528, 1, 672, 3]}>

    %output = VPURT.DeclareBuffer "DDR" <0> -> !Output_DDR
    %output0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output1 = VPURT.DeclareBuffer "DDR" <3> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output2 = VPURT.DeclareBuffer "DDR" <6> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output3 = VPURT.DeclareBuffer "DDR" <9> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output4 = VPURT.DeclareBuffer "DDR" <12> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output5 = VPURT.DeclareBuffer "DDR" <15> -> memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output0: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output1: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) attributes {cycleBegin = 3 : i64, cycleEnd = 4 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output2: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) attributes {cycleBegin = 4 : i64, cycleEnd = 5 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output3: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4 : !VPURT.Barrier) attributes {cycleBegin = 5 : i64, cycleEnd = 6 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output4: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar4 : !VPURT.Barrier) attributes {cycleBegin = 6 : i64, cycleEnd = 7 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input0 : memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [150528, 1, 672, 3]}>) outputs(%output5: memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    return %output : !Output_DDR


    // CHECK-NOT:   VPURT.DeclareVirtualBarrier

    // CHECK:    VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task attributes {cycleBegin = 3 : i64, cycleEnd = 4 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task attributes {cycleBegin = 4 : i64, cycleEnd = 5 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task attributes {cycleBegin = 5 : i64, cycleEnd = 6 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task attributes {cycleBegin = 6 : i64, cycleEnd = 7 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!Input_DDR = type memref<1x3x224x224x!qElemType0, #NHWC, @DDR>
!Output_DDR = type memref<1x16x224x224x!qElemType0, #NHWC, @DDR>

//CHECK-LABEL: @NoDMABarrierOptimization
func @NoDMABarrierOptimization() -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %input0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [150528, 1, 672, 3]}>

    %output = VPURT.DeclareBuffer "DDR" <0> -> !Output_DDR
    %output0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output1 = VPURT.DeclareBuffer "DDR" <3> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output2 = VPURT.DeclareBuffer "DDR" <6> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output3 = VPURT.DeclareBuffer "DDR" <9> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output4 = VPURT.DeclareBuffer "DDR" <12> -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output5 = VPURT.DeclareBuffer "DDR" <15> -> memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output0: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input : !Input_DDR) outputs(%output1: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) attributes {cycleBegin = 3 : i64, cycleEnd = 4 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output2: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) attributes {cycleBegin = 4 : i64, cycleEnd = 5 : i64} {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input : !Input_DDR) outputs(%output3: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4 : !VPURT.Barrier) attributes {cycleBegin = 5 : i64, cycleEnd = 6 : i64} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !Input_DDR) outputs(%output4: memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar4 : !VPURT.Barrier) attributes {cycleBegin = 6 : i64, cycleEnd = 7 : i64} {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input0 : memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [150528, 1, 672, 3]}>) outputs(%output5: memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x1x224x224x!qElemType0, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    return %output : !Output_DDR


    // CHECK:     [[Bar0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[Bar0]] : !VPURT.Barrier) attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar0]] : !VPURT.Barrier) updates([[Bar1]] : !VPURT.Barrier) attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar1]] : !VPURT.Barrier) updates([[Bar2]] : !VPURT.Barrier) attributes {cycleBegin = 3 : i64, cycleEnd = 4 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar2]] : !VPURT.Barrier) updates([[Bar3]] : !VPURT.Barrier) attributes {cycleBegin = 4 : i64, cycleEnd = 5 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar3]] : !VPURT.Barrier) updates([[Bar4]] : !VPURT.Barrier) attributes {cycleBegin = 5 : i64, cycleEnd = 6 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar4]] : !VPURT.Barrier) attributes {cycleBegin = 6 : i64, cycleEnd = 7 : i64} {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}
