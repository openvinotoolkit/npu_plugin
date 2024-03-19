//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --hardware-adaptation %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

 // CHECK: func.func @TwoDMAs([[ARG0:%.*]]: memref<10xf16>, [[ARG1:%.*]]: memref<5xf16>) -> memref<5xf16> {
func.func @TwoDMAs(%arg0: memref<10xf16>, %arg1: memref<5xf16>) -> memref<5xf16> {
    %in_subview = VPUIP.SubView %arg0 [0] [5] : memref<10xf16> to memref<5xf16>
    %buf0 = VPUIP.StaticAlloc<0> -> memref<5xf16, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<5xf16, [@CMX_NN, 0]>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %0 = VPUIP.Copy inputs(%in_subview : memref<5xf16>) outputs(%buf0 : memref<5xf16, [@CMX_NN, 0]>) -> memref<5xf16, [@CMX_NN, 0]>
        async.yield %buf0 : memref<5xf16, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute[%t0] (%f0 as %0: !async.value<memref<5xf16, [@CMX_NN, 0]>>) -> !async.value<memref<5xf16>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %1 = VPUIP.Copy inputs(%buf0 : memref<5xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<5xf16>) -> memref<5xf16>
        async.yield %arg1: memref<5xf16>
    }

    %1 = async.await %f1 : !async.value<memref<5xf16>>
    return %1 : memref<5xf16>

    // CHECK:       [[IN_SUBVIEW:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<5xf16>
    // CHECK:       [[IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<5xf16, [@CMX_NN, 0]>
    // CHECK:       [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
    // CHECK:           %3 = VPUIP.Copy inputs([[IN_SUBVIEW]] : memref<5xf16>)
    // CHECK-SAME:                                        outputs([[IN_CMX]] : memref<5xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:           -> memref<5xf16, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       VPURT.Task waits([[BAR]] : !VPURT.Barrier) {
    // CHECK:           %3 = VPUIP.Copy inputs([[IN_CMX]] : memref<5xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:                                        outputs([[ARG1]] : memref<5xf16>)
    // CHECK-SAME:           -> memref<5xf16>
    // CHECK:       }

    // CHECK:       return [[ARG1]] : memref<5xf16>
}
