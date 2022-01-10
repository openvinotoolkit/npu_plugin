// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s

// CHECK-LABEL: @ControlEdgeOverlapMemoryCheckProdCons
module @ControlEdgeOverlapMemoryCheckProdCons {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x120000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x120000xf16>
    }

func @main(%in: memref<1x120000xf16>, %out: memref<1x120000xf16>) -> memref<1x120000xf16> {
    %cst0 = const.Declare memref<1x120000xf16> = #const.Content<dense<2.0> : tensor<1x120000xf16>>

    %buf_in = memref.alloc() : memref<1x120000xf16, @CMX_NN>

    %buf0 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x120000xf16, @CMX_NN>

    %t_in, %r_in = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%in : memref<1x120000xf16>) outputs(%buf_in : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %1 = IERT.ReLU inputs(%0: memref<1x120000xf16, @CMX_NN>) outputs(%buf0 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %1 : memref<1x120000xf16, @CMX_NN>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf1 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r0 as %1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, @CMX_NN>, %1: memref<1x120000xf16, @CMX_NN>) outputs(%buf2 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %2 : memref<1x120000xf16, @CMX_NN>
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r1 as %1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, @CMX_NN>, %1: memref<1x120000xf16, @CMX_NN>) outputs(%buf3 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %2 : memref<1x120000xf16, @CMX_NN>
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16>> {
        %1 = IERT.Copy inputs(%0 : memref<1x120000xf16, @CMX_NN>) outputs(%out : memref<1x120000xf16>) -> memref<1x120000xf16>
        async.yield %1 : memref<1x120000xf16>
    }

    %6 = async.await %r6 : !async.value<memref<1x120000xf16>>
    return %6 : memref<1x120000xf16>

    // Token dependencies will match data flow by default:
    //  Task0 -> Task1 -> Task3 -> Task4
    //  Task2 -> Task3
    //  Task0 -> Task_SW -> Task_SR -> Task4 -> Task5
    // besides that due to overlapping memory ranges additional control edge will be inserted.
    // Important is relation between Task0, Task1, Task_SW.
    // Execution order is following:
    //  t0: Task0 produces BUF0
    //  tX: Task1 reads BUF0
    //  tY: Task_SW reads BUF0
    // Resulting dependencies from just looking at memory intervals and their users throughout execution time
    // is following: Task0 -> Task1, Task0 -> Task_SW
    // If there would be no differentiation between resource producer and consumer unnecessary dependency
    // would be inserted from Task1 -> Task_SW
    //
    // Optimization of token dependencies (transitive reduction) is beyond
    // this pass and done as a separate step

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<0>
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<240000>
    // CHECK:       [[BUF2:%.*]] = IERT.StaticAlloc<480000>
    // CHECK:       [[BUF3:%.*]] = IERT.StaticAlloc<0>
    // CHECK:       [[BUF4:%.*]] = IERT.StaticAlloc<480000>
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x120000xf16, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = IERT.StaticAlloc<240000> -> memref<1x120000xf16, @CMX_NN>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       IERT.Copy
    // CHECK-SAME:       outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:       [[T0]]
    // CHECK-NEXT:       IERT.ReLU
    // CHECK-SAME:       outputs([[BUF1]]

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute ->
    // CHECK-NEXT:       IERT.Copy
    // CHECK-SAME:       outputs([[BUF2]]

    // CHECK:       [[T_SW:%.+]], [[R_SW:%.+]] = async.execute
    // CHECK-SAME:       [[T0]]
    // CHECK-NOT:        [[T1]]
    // CHECK-NEXT:       IERT.Copy
    // CHECK-SAME:       inputs([[BUF0]]
    // CHECK-SAME:       outputs([[BUF_SPILL_WRITE]]

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:       [[T1]], [[T2]], [[T_SW]]
    // CHECK-NEXT:       IERT.Add
    // CHECK-SAME:       outputs([[BUF3]]

    // CHECK:       [[T_SR:%.+]], [[R_SR:%.+]] = async.execute
    // CHECK-SAME:       [[T3]], [[T_SW]]
    // CHECK-NEXT:       IERT.Copy
    // CHECK-SAME:       outputs([[BUF_SPILL_READ]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:       [[T0]], [[T3]], [[T_SR]]
    // CHECK-NEXT:       IERT.Add
    // CHECK-SAME:       outputs([[BUF4]]

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:       [[T4]]
    // CHECK-NEXT:       IERT.Copy
}

}
