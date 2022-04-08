//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --dma-task-profiling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
    IE.MemoryResource 1048576 bytes of @CMX_NN
}

!defaultType = type memref<1x16x62x62xf16>
!DdrType = type memref<1x16x62x62xf16, @DDR>
!PermuteCmxType = type memref<1x16x62x62xf16, #NHWC, @CMX_NN>
!PermuteDdrType = type memref<1x16x62x62xf16, #NHWC, @DDR>


IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "in" : tensor<1x16x62x62xf16>
} outputsInfo :  {
    DataInfo "out" : tensor<1x16x62x62xf16>
} profilingOutputsInfo :  {
}
func @main(%arg0: !defaultType, %arg1: !PermuteDdrType) -> !PermuteDdrType {
    %0 = memref.alloc() : !DdrType
    %1 = memref.alloc() : !PermuteCmxType
    %token_0, %results_0 = async.execute -> !async.value<!DdrType> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %2 = VPUIP.Copy inputs(%arg0 : !defaultType) outputs(%0 : !DdrType) -> !DdrType
        async.yield %2 : !DdrType
    }
    %token_1, %results_1 = async.execute [%token_0] (%results_0 as %arg2: !async.value<!DdrType>)-> !async.value<!PermuteCmxType> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %2 = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs(%arg2 : !DdrType) outputs(%1 : !PermuteCmxType) -> !PermuteCmxType
        async.yield %2 : !PermuteCmxType
    }
    %token_2, %results_2 = async.execute [%token_1] (%results_1 as %arg2: !async.value<!PermuteCmxType>)-> !async.value<!PermuteDdrType> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %2 = VPUIP.Copy inputs(%arg2 : !PermuteCmxType) outputs(%arg1 : !PermuteDdrType) -> !PermuteDdrType
        async.yield %2 : !PermuteDdrType
    }
    %3 = async.await %results_2 : !async.value<!PermuteDdrType>
    return %3 : !PermuteDdrType
}

//CHECK:        profilingOutputsInfo
//CHECK-NEXT:   DataInfo "dma" : tensor<6xui64>
//CHECK:        func @main(%arg0: memref<1x16x62x62xf16>, %arg1: memref<1x16x62x62xf16, #NHWC, @DDR>, %arg2: memref<6xui64>) -> (memref<1x16x62x62xf16, #NHWC, @DDR>, memref<6xui64>) {
//CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<6xui64, [@CMX_NN, 0]>

//CHECK:        async.execute
//CHECK-NEXT:   [[VAR1:%.+]] = VPUIP.SubView [[VAR0]] [0] [1] : memref<6xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR2:%.+]] = VPUIP.Timestamp([[VAR1]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR3:%.+]] = VPUIP.Copy
//CHECK-NEXT:   [[VAR4:%.+]] = VPUIP.SubView [[VAR0]] [1] [1] : memref<6xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR5:%.+]] = VPUIP.Timestamp([[VAR4]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   async.yield [[VAR3]], [[VAR2]], [[VAR5]] : memref<1x16x62x62xf16, @DDR>, memref<1xui64, [@CMX_NN, 0]>, memref<1xui64, [@CMX_NN, 0]>

//CHECK:        async.execute
//CHECK-NEXT:   [[VAR6:%.+]] = VPUIP.SubView [[VAR0]] [2] [1] : memref<6xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR7:%.+]] = VPUIP.Timestamp([[VAR6]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR8:%.+]] = VPUIP.PermuteDMA
//CHECK-NEXT:   [[VAR9:%.+]] = VPUIP.SubView [[VAR0]] [3] [1] : memref<6xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR10:%.+]] = VPUIP.Timestamp([[VAR9]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   async.yield [[VAR8]], [[VAR7]], [[VAR10]] : memref<1x16x62x62xf16, #NHWC, @CMX_NN>, memref<1xui64, [@CMX_NN, 0]>, memref<1xui64, [@CMX_NN, 0]>

//CHECK:        async.execute
//CHECK-NEXT:   [[VAR11:%.+]] = VPUIP.SubView [[VAR0]] [4] [1] : memref<6xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR12:%.+]] = VPUIP.Timestamp([[VAR11]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR13:%.+]] = VPUIP.Copy
//CHECK-NEXT:   [[VAR14:%.+]] = VPUIP.SubView [[VAR0]] [5] [1] : memref<6xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR15:%.+]] = VPUIP.Timestamp([[VAR14]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   async.yield [[VAR13]], [[VAR2]], [[VAR15]] : memref<1x16x62x62xf16, #NHWC, @DDR>, memref<1xui64, [@CMX_NN, 0]>, memref<1xui64, [@CMX_NN, 0]>

//CHECK:        [[token_0:%.*]], [[result_0:%.*]] = async.execute
//CHECK-NEXT:   [[VAR16:%.+]] = VPUIP.SubView %arg2 [0] [6] : memref<6xui64> to memref<6xui64>
//CHECK-NEXT:   [[VAR17:%.+]] = VPUIP.ConcatView
//CHECK-NEXT:   [[VAR18:%.+]] = VPUIP.Copy inputs([[VAR17]] : memref<6xui64, [@CMX_NN, 0]>) outputs([[VAR16]] : memref<6xui64>) -> memref<6xui64>
//CHECK-NEXT:   async.yield [[VAR18]] : memref<6xui64>

//CHECK:        [[VAR19:%.+]] = async.await [[result_0]] : !async.value<memref<6xui64>>
//CHECK:        [[VAR20:%.+]] = async.await
//CHECK:        [[VAR21:%.+]] = VPUIP.ConcatView inputs([[VAR19]] : memref<6xui64>) outputs(%arg2 : memref<6xui64>) -> memref<6xui64>
//CHECK-NEXT:   return [[VAR20]], [[VAR21]] : memref<1x16x62x62xf16, #NHWC, @DDR>, memref<6xui64>
