//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
!MemRef1 = memref<1x128x64x32xf16, #NWHC>
!Distributed0 = !VPUIP.DistributedBuffer<1x128x64x32xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed1 = !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!Distributed2 = !VPUIP.DistributedBuffer<1x62x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
!MemRef0 = memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>
!MemRef2 = memref<1x62x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, @CMX_NN>

func.func @ParsePrintDistributedBuffer(%arg0: !MemRef1) -> !MemRef1 {
    %0 = VPURT.AllocDistributed -> !Distributed0
    %1 = VPURT.AllocDistributed -> !Distributed0
    %2 = memref.alloc() : !MemRef1
    %token, %results = async.execute -> !async.value<!Distributed0> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64} {
        %4 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !MemRef1) outputs(%0 as %arg2: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> !Distributed0 {
        %5 = VPUIP.Copy inputs(%arg1 : !MemRef1) outputs(%arg2 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) -> memref<1x128x64x32xf16, #NWHC, @CMX_NN>
        }
        async.yield %4 : !Distributed0
    }
    %token_0, %results_1:2 = async.execute [%token] (%results as %arg1: !async.value<!Distributed0>) -> (!async.value<!Distributed1>, !async.value<!Distributed2>) attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %4 = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 62, 64, 32] : !Distributed0 to !Distributed2
        %5 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 64, 64, 32] : !Distributed0 to !Distributed1
        %6 = VPUIP.SubView %1 [0, 62, 0, 0] [1, 62, 64, 32] : !Distributed0 to !Distributed2
        %7 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 64, 64, 32] : !Distributed0 to !Distributed1
        %8:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN
            inputs(%5 as %arg6: !Distributed1, %4 as %arg7: !Distributed2)
            outputs(%7 as %arg8: !Distributed1, %6 as %arg9: !Distributed2) on tile 0 -> (!Distributed1, !Distributed2){
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : !Distributed1, !Distributed1
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg7, %arg9) : !Distributed2, !Distributed2
        }
        async.yield %8#0, %8#1 : !Distributed1, !Distributed2
    }
    %token_2, %results_3 = async.execute [%token_0] (%results_1#0 as %arg1: !async.value<!Distributed1>, %results_1#1 as %arg2: !async.value<!Distributed2>) -> !async.value<!MemRef1> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64} {
        %4 = VPUIP.ConcatView inputs(%arg1, %arg2 : !Distributed1, !Distributed2) outputs(%1 : !Distributed0) -> !Distributed0
        %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg3: memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%2 as %arg4: !MemRef1) -> !MemRef1 {
        %6 = VPUIP.Copy inputs(%arg3 : memref<1x128x64x32xf16, #NWHC, @CMX_NN>) outputs(%arg4 : !MemRef1) -> !MemRef1
        }
        async.yield %5 : !MemRef1
    }
    %3 = async.await %results_3 : !async.value<!MemRef1>

    return %3 : !MemRef1

    //CHECK:        %token_0, %results_1:2 = async.execute [%token]
    //CHECK-SAME:        attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    //CHECK:                %results_4:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN
    //CHECK-SAME:               inputs(%5 as %arg2: !VPUIP.DistributedBuffer
    //CHECK-SAME:                      %4 as %arg3: !VPUIP.DistributedBuffer
    //CHECK-SAME:               outputs(%7 as %arg4: !VPUIP.DistributedBuffer
    //CHECK-SAME:                       %6 as %arg5: !VPUIP.DistributedBuffer
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : !VPUIP.DistributedBuffer
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg5) : !VPUIP.DistributedBuffer
    //CHECK:              async.yield %results_4#0, %results_4#1

}
