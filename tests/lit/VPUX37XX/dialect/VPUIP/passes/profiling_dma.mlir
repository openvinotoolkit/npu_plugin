// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --dma-task-profiling %s | FileCheck %s

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
    IE.MemoryResource 1048576 bytes of @CMX_NN
}

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "in" : tensor<1x16x62x62xf16>
} outputsInfo :  {
    DataInfo "out" : tensor<1x16x62x62xf16>
} profilingOutputsInfo :  {
}
func @main(%arg0: memref<1x16x62x62xf16>, %arg1: memref<1x16x62x62xf16>) -> memref<1x16x62x62xf16> {
    %0 = memref.alloc() : memref<1x16x62x62xf16, @DDR>
    %token_0, %results_0 = async.execute -> !async.value<memref<1x16x62x62xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x62x62xf16>) outputs(%0 : memref<1x16x62x62xf16, @DDR>) -> memref<1x16x62x62xf16, @DDR>
        async.yield %1 : memref<1x16x62x62xf16, @DDR>
    }
    %token_1, %results_1 = async.execute [%token_0] (%results_0 as %arg2: !async.value<memref<1x16x62x62xf16, @DDR>>)-> !async.value<memref<1x16x62x62xf16>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.Copy inputs(%arg2 : memref<1x16x62x62xf16, @DDR>) outputs(%arg1 : memref<1x16x62x62xf16>) -> memref<1x16x62x62xf16>
        async.yield %1 : memref<1x16x62x62xf16>
    }
    %2 = async.await %results_1 : !async.value<memref<1x16x62x62xf16>>
    return %2 : memref<1x16x62x62xf16>
}

//CHECK:        profilingOutputsInfo
//CHECK-NEXT:   DataInfo "dma" : tensor<4xui64>
//CHECK:        func @main(%arg0: memref<1x16x62x62xf16>, %arg1: memref<1x16x62x62xf16>, %arg2: memref<4xui64>) -> (memref<1x16x62x62xf16>, memref<4xui64>) {
//CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<4xui64, [@CMX_NN, 0]>

//CHECK:        async.execute
//CHECK-NEXT:   [[VAR1:%.+]] = VPUIP.SubView [[VAR0]] [0] [1] : memref<4xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR2:%.+]] = VPUIP.Timestamp([[VAR1]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR3:%.+]] = VPUIP.Copy
//CHECK-NEXT:   [[VAR4:%.+]] = VPUIP.SubView [[VAR0]] [1] [1] : memref<4xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR5:%.+]] = VPUIP.Timestamp([[VAR4]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   async.yield [[VAR3]], [[VAR2]], [[VAR5]] : memref<1x16x62x62xf16, @DDR>, memref<1xui64, [@CMX_NN, 0]>, memref<1xui64, [@CMX_NN, 0]>

//CHECK:        async.execute
//CHECK-NEXT:   [[VAR6:%.+]] = VPUIP.SubView [[VAR0]] [2] [1] : memref<4xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR7:%.+]] = VPUIP.Timestamp([[VAR6]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR8:%.+]] = VPUIP.Copy
//CHECK-NEXT:   [[VAR9:%.+]] = VPUIP.SubView [[VAR0]] [3] [1] : memref<4xui64, [@CMX_NN, 0]> to memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   [[VAR10:%.+]] = VPUIP.Timestamp([[VAR9]] : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]>
//CHECK-NEXT:   async.yield [[VAR8]], [[VAR7]], [[VAR10]] : memref<1x16x62x62xf16>, memref<1xui64, [@CMX_NN, 0]>, memref<1xui64, [@CMX_NN, 0]>

//CHECK:        [[token_0:%.*]], [[result_0:%.*]] = async.execute
//CHECK-NEXT:   [[VAR16:%.+]] = VPUIP.SubView %arg2 [0] [4] : memref<4xui64> to memref<4xui64>
//CHECK-NEXT:   [[VAR13:%.+]] = VPUIP.ConcatView
//CHECK-NEXT:   [[VAR14:%.+]] = VPUIP.Copy inputs([[VAR13]] : memref<4xui64, [@CMX_NN, 0]>) outputs([[VAR16]] : memref<4xui64>) -> memref<4xui64>
//CHECK-NEXT:   async.yield [[VAR14]] : memref<4xui64>
//CHECK:        [[VAR15:%.+]] = async.await [[result_0]] : !async.value<memref<4xui64>>
//CHECK:        [[VAR12:%.+]] = async.await
//CHECK:        [[VAR17:%.+]] = VPUIP.ConcatView inputs([[VAR15]] : memref<4xui64>) outputs(%arg2 : memref<4xui64>) -> memref<4xui64>
//CHECK-NEXT:   return [[VAR12]], [[VAR17]] : memref<1x16x62x62xf16>, memref<4xui64>
