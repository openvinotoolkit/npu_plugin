// RUN: vpux-opt --dma-task-profiling %s | FileCheck %s

// CHECK-LABEL: @DmaProfiling
module @DmaProfiling {

    IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
    IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
    IE.MemoryResource 1048576 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

    module @UsedMemory {
        IE.MemoryResource 2048 bytes of @DDR
        IE.MemoryResource 1048576 bytes of @CMX_NN
    }

    IE.ExecutorResource 16 of @SHAVE_UPA
    IE.ExecutorResource 4 of  @NCE {
        IE.ExecutorResource 5 of @DPU
    }
    IE.ExecutorResource 1 of @DMA_NN

    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "in" : tensor<1x16x62x62xf16>
    } outputsInfo :  {
        DataInfo "out" : tensor<1x16x62x62xf16>
    } profilingOutputsInfo :  {
    }
    func @main(%arg0: memref<1x16x62x62xf16>, %arg1: memref<1x16x62x62xf16>) -> memref<1x16x62x62xf16> {
        %0 = memref.alloc() : memref<1x16x62x62xf16, @DDR>
        %token_0, %results_0 = async.execute -> !async.value<memref<1x16x62x62xf16, @DDR>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
            %1 = IERT.Copy inputs(%arg0 : memref<1x16x62x62xf16>) outputs(%0 : memref<1x16x62x62xf16, @DDR>) -> memref<1x16x62x62xf16, @DDR>
            async.yield %1 : memref<1x16x62x62xf16, @DDR>
        }
        %token_1, %results_1 = async.execute [%token_0] (%results_0 as %arg2: !async.value<memref<1x16x62x62xf16, @DDR>>)-> !async.value<memref<1x16x62x62xf16>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
            %1 = IERT.Copy inputs(%arg2 : memref<1x16x62x62xf16, @DDR>) outputs(%arg1 : memref<1x16x62x62xf16>) -> memref<1x16x62x62xf16>
            async.yield %1 : memref<1x16x62x62xf16>
        }
        %2 = async.await %results_1 : !async.value<memref<1x16x62x62xf16>>
        return %2 : memref<1x16x62x62xf16>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "dma" : tensor<4xui32>
    //CHECK:        func @main(%arg0: memref<1x16x62x62xf16>, %arg1: memref<1x16x62x62xf16>, %arg2: memref<4xui32>) -> (memref<1x16x62x62xf16>, memref<4xui32>) {
    //CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<4xui32, [@CMX_NN, 0]>

    //CHECK:        async.execute
    //CHECK-NEXT:   [[VAR1:%.+]] = IERT.SubView [[VAR0]] [0] [1] : memref<4xui32, [@CMX_NN, 0]> to memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   [[VAR2:%.+]] = IERT.Timestamp([[VAR1]] : memref<1xui32, [@CMX_NN, 0]>) -> memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   [[VAR3:%.+]] = IERT.Copy
    //CHECK-NEXT:   [[VAR4:%.+]] = IERT.SubView [[VAR0]] [1] [1] : memref<4xui32, [@CMX_NN, 0]> to memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   [[VAR5:%.+]] = IERT.Timestamp([[VAR4]] : memref<1xui32, [@CMX_NN, 0]>) -> memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   async.yield [[VAR3]], [[VAR2]], [[VAR5]] : memref<1x16x62x62xf16, @DDR>, memref<1xui32, [@CMX_NN, 0]>, memref<1xui32, [@CMX_NN, 0]>

    //CHECK:        async.execute
    //CHECK-NEXT:   [[VAR6:%.+]] = IERT.SubView [[VAR0]] [2] [1] : memref<4xui32, [@CMX_NN, 0]> to memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   [[VAR7:%.+]] = IERT.Timestamp([[VAR6]] : memref<1xui32, [@CMX_NN, 0]>) -> memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   [[VAR8:%.+]] = IERT.Copy
    //CHECK-NEXT:   [[VAR9:%.+]] = IERT.SubView [[VAR0]] [3] [1] : memref<4xui32, [@CMX_NN, 0]> to memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   [[VAR10:%.+]] = IERT.Timestamp([[VAR9]] : memref<1xui32, [@CMX_NN, 0]>) -> memref<1xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   async.yield [[VAR8]], [[VAR7]], [[VAR10]] : memref<1x16x62x62xf16>, memref<1xui32, [@CMX_NN, 0]>, memref<1xui32, [@CMX_NN, 0]>

    //CHECK:        [[token_0:%.*]], [[result_0:%.*]] = async.execute
    //CHECK-NEXT:   [[VAR16:%.+]] = IERT.SubView %arg2 [0] [4] : memref<4xui32> to memref<4xui32>
    //CHECK-NEXT:   [[VAR13:%.+]] = IERT.ConcatView
    //CHECK-NEXT:   [[VAR14:%.+]] = IERT.Copy inputs([[VAR13]] : memref<4xui32, [@CMX_NN, 0]>) outputs([[VAR16]] : memref<4xui32>) -> memref<4xui32>
    //CHECK-NEXT:   async.yield [[VAR14]] : memref<4xui32>
    //CHECK:        [[VAR15:%.+]] = async.await [[result_0]] : !async.value<memref<4xui32>>
    //CHECK:        [[VAR12:%.+]] = async.await
    //CHECK:        [[VAR17:%.+]] = IERT.ConcatView inputs([[VAR15]] : memref<4xui32>) outputs(%arg2 : memref<4xui32>) -> memref<4xui32>
    //CHECK-NEXT:   return [[VAR12]], [[VAR17]] : memref<1x16x62x62xf16>, memref<4xui32>
}
