// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s

// CHECK-LABEL: @SimpleGraph
module @SimpleGraph {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1000xf16>
    }

// CHECK:   module @UsedMemory
// CHECK:           IE.MemoryResource 4096 bytes of @CMX_NN

func @main(%in: memref<1x1000xf16>, %out: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %buf0 = memref.alloc() : memref<1x1000xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x1000xf16, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x1000xf16, @CMX_NN>

    %t0, %f0 = async.execute -> !async.value<memref<1x1000xf16, @CMX_NN>> {
        %0 = IERT.ReLU inputs(%in : memref<1x1000xf16>) outputs(%buf0 : memref<1x1000xf16, @CMX_NN>) -> memref<1x1000xf16, @CMX_NN>
        async.yield %0 : memref<1x1000xf16, @CMX_NN>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x1000xf16, @CMX_NN>>)
            -> !async.value<memref<1x1000xf16, @CMX_NN>> {
        %1 = IERT.ReLU inputs(%0: memref<1x1000xf16, @CMX_NN>) outputs(%buf1 : memref<1x1000xf16, @CMX_NN>) -> memref<1x1000xf16, @CMX_NN>
        async.yield %1 : memref<1x1000xf16, @CMX_NN>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1000xf16, @CMX_NN>>)
            -> !async.value<memref<1x1000xf16, @CMX_NN>> {
        %2 = IERT.ReLU inputs(%1: memref<1x1000xf16, @CMX_NN>) outputs(%buf2 : memref<1x1000xf16, @CMX_NN>) -> memref<1x1000xf16, @CMX_NN>
        async.yield %2 : memref<1x1000xf16, @CMX_NN>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x1000xf16, @CMX_NN>>)
            -> !async.value<memref<1x1000xf16>> {
        %3 = IERT.Copy inputs(%2 : memref<1x1000xf16, @CMX_NN>) outputs(%out : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %3 : memref<1x1000xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x1000xf16>>
    return %3 : memref<1x1000xf16>

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, @CMX_NN>
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<2048> -> memref<1x1000xf16, @CMX_NN>
    // CHECK:       [[BUF2:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, @CMX_NN>

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x1000xf16, @CMX_NN>)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x1000xf16, @CMX_NN>)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x1000xf16, @CMX_NN>)

    // CHECK:       IERT.Copy
}

}

// -----

// CHECK-LABEL: @TwoOutputs
module @TwoOutputs {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<2xf16>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<2xf16>
        DataInfo "prob2" : tensor<2xf16>
    }

// CHECK:   module @UsedMemory
// CHECK:           IE.MemoryResource 128 bytes of @CMX_NN

func @main(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %cst = const.Declare memref<2xf16, @CMX_NN> = #const.Content<dense<1.000000e+00> : tensor<2xf16>>
    %buf0 = memref.alloc() : memref<2xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<2xf16, @CMX_NN>
    %token, %results = async.execute -> !async.value<memref<2xf16, @CMX_NN>> attributes {"async-deps-index" = 0 : i64} {
      %4 = IERT.ReLU inputs(%arg0 : memref<2xf16>) outputs(%buf0 : memref<2xf16, @CMX_NN>) -> memref<2xf16, @CMX_NN>
      async.yield %4 : memref<2xf16, @CMX_NN>
    }
    %token_0, %results_1 = async.execute -> !async.value<memref<2xf16, @CMX_NN>> attributes {"async-deps-index" = 1 : i64} {
      %4 = IERT.ReLU inputs(%cst : memref<2xf16, @CMX_NN>) outputs(%buf1 : memref<2xf16, @CMX_NN>) -> memref<2xf16, @CMX_NN>
      async.yield %4 : memref<2xf16, @CMX_NN>
    }
    %token_2, %results_3 = async.execute [%token] (%results as %arg3: !async.value<memref<2xf16, @CMX_NN>>) -> !async.value<memref<2xf16>> attributes {"async-deps-index" = 2 : i64} {
      %4 = IERT.Copy inputs(%arg3 : memref<2xf16, @CMX_NN>) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
      async.yield %4 : memref<2xf16>
    }
    %token_4, %results_5 = async.execute [%token_0] (%results_1 as %arg3: !async.value<memref<2xf16, @CMX_NN>>) -> !async.value<memref<2xf16>> attributes {"async-deps-index" = 3 : i64} {
      %4 = IERT.Copy inputs(%arg3 : memref<2xf16, @CMX_NN>) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
      async.yield %4 : memref<2xf16>
    }
    %2 = async.await %results_3 : !async.value<memref<2xf16>>
    %3 = async.await %results_5 : !async.value<memref<2xf16>>
    return %2, %3 : memref<2xf16>, memref<2xf16>

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<64> -> memref<2xf16, @CMX_NN>
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<0> -> memref<2xf16, @CMX_NN>

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF1]] : memref<2xf16, @CMX_NN>)

    // CHECK:       IERT.Copy

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF0]] : memref<2xf16, @CMX_NN>)

    // CHECK:       IERT.Copy
}

}

// -----

// CHECK-LABEL: @Spilling
module @Spilling {

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

    %t0, %r0 = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf0 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %1 = IERT.ReLU inputs(%0: memref<1x120000xf16, @CMX_NN>) outputs(%buf1 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %1 : memref<1x120000xf16, @CMX_NN>
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

    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x120000xf16, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = IERT.StaticAlloc<
    // CHECK-SAME:      > -> memref<1x120000xf16, @CMX_NN>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       IERT.Copy inputs(%arg0 : memref<1x120000xf16>) outputs([[BUF0:%.*]] :

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       IERT.ReLU

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy inputs([[BUF0]] : memref<1x120000xf16, @CMX_NN>) outputs([[BUF_SPILL_WRITE]] : memref<1x120000xf16, @DDR>)

    // CHECK:       [[T22:%.+]], [[R22:%.+]] = async.execute
    // CHECK:       IERT.Add

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      ([[R3]] as [[ARG1:%.*]]: !async.value<memref<1x120000xf16, @DDR>>
    // CHECK:       IERT.Copy inputs([[ARG1:%.*]] : memref<1x120000xf16, @DDR>) outputs([[BUF_SPILL_READ]] : memref<1x120000xf16, @CMX_NN>)

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      ([[R4]] as [[ARG2:%.*]]: !async.value<memref<1x120000xf16, @CMX_NN>>,
    // CHECK:       IERT.Add inputs
    // CHECK-SAME:      [[ARG2]] : memref<1x120000xf16, @CMX_NN>

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy
}

}

// -----

// CHECK-LABEL: @SpillingOpWith2Outputs
module @SpillingOpWith2Outputs {

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

    %buf0 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf5 = memref.alloc() : memref<1x120000xf16, @CMX_NN>

    %t0, %r0 = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%in : memref<1x120000xf16>) outputs(%buf0 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t1, %r1 = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf3 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    // Below operation has two outputs and one of them would need to be spilled when scheduled
    %t2, %r2:2 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> (!async.value<memref<1x120000xf16, @CMX_NN>>, !async.value<memref<1x120000xf16, @CMX_NN>>) {
        %1 = IERT.ReLU inputs(%arg0: memref<1x120000xf16, @CMX_NN>) outputs(%buf1 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        %2 = IERT.ReLU inputs(%arg0: memref<1x120000xf16, @CMX_NN>) outputs(%buf2 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %1, %2 : memref<1x120000xf16, @CMX_NN>, memref<1x120000xf16, @CMX_NN>
    }

    %t3, %r3 = async.execute [%t1, %t2] (%r2#0 as %arg0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r1 as %arg1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Add inputs(%arg0: memref<1x120000xf16, @CMX_NN>, %arg1: memref<1x120000xf16, @CMX_NN>) outputs(%buf4: memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t4, %r4 = async.execute [%t1, %t3] (%r2#1 as %arg0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r3 as %arg1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Add inputs(%arg0: memref<1x120000xf16, @CMX_NN>, %arg1: memref<1x120000xf16, @CMX_NN>) outputs(%buf5: memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t5, %r5 = async.execute [%t4] (%r4 as %arg0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16>> {
        %0 = IERT.Copy inputs(%arg0 : memref<1x120000xf16, @CMX_NN>) outputs(%out : memref<1x120000xf16>) -> memref<1x120000xf16>
        async.yield %0 : memref<1x120000xf16>
    }

    %3 = async.await %r5 : !async.value<memref<1x120000xf16>>
    return %3 : memref<1x120000xf16>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT       IERT.Copy
    // CHECK:       [[T1:%.+]], [[R1:%.+]]:2 = async.execute {{.*}} ([[R0]] as %arg2: !async.value<memref<1x120000xf16, @CMX_NN>>)
    // CHECK-NEXT       IERT.ReLU inputs(%arg2 : memref<1x120000xf16, @CMX_NN>) outputs([[BUF1:%.*]] :
    // CHECK-NEXT       IERT.ReLU inputs(%arg2 : memref<1x120000xf16, @CMX_NN>) outputs([[BUF2:%.*]] :
    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT       IERT.Copy
    // CHECK:       [[T_SPILL_WRITE:%.+]], [[R_SPILL_WRITE:%.+]] = async.execute
    // CHECK-NEXT       IERT.Copy inputs([[BUF2]] : memref<1x120000xf16, @CMX_NN>)
    // CHECK-SAME       -> memref<1x120000xf16, @DDR>
    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute {{.*}} ([[R1]]#0 as %arg2: !async.value<memref<1x120000xf16, @CMX_NN>>
    // CHECK-NEXT       IERT.Add inputs(%arg2 : memref<1x120000xf16, @CMX_NN>)
    // CHECK:       [[T_SPILL_READ:%.+]], [[R_SPILL_READ:%.+]] = async.execute {{.*}} ([[R_SPILL_WRITE]] as %arg2: !async.value<memref<1x120000xf16, @DDR>>)
    // CHECK-NEXT       IERT.Copy inputs(%arg2 : memref<1x120000xf16, @DDR>)
    // CHECK-SAME       -> memref<1x120000xf16, @CMX_NN>
    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute {{.*}} ([[R_SPILL_READ]] as %arg2: !async.value<memref<1x120000xf16, @CMX_NN>>
    // CHECK-NEXT       IERT.Add inputs(%arg2 : memref<1x120000xf16, @CMX_NN>)
    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute {{.*}} ([[R6]] as %arg2: !async.value<memref<1x120000xf16, @CMX_NN>>
    // CHECK-NEXT       IERT.Copy
}

}

// -----

// CHECK-LABEL: @SpillingOfSubViewBuffer
module @SpillingOfSubViewBuffer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<100000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<100000xf16>
    }

func @main(%in: memref<100000xf16>, %out: memref<100000xf16>) -> memref<100000xf16> {
    %cst0 = const.Declare memref<100000xf16> = #const.Content<dense<2.0> : tensor<100000xf16>>

    // master buffer that will get spilled
    %buf_master = memref.alloc() : memref<200000xf16, @CMX_NN>

    %buf0 = memref.alloc() : memref<100000xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<100000xf16, @CMX_NN>
    %buf2 = memref.alloc() : memref<100000xf16, @CMX_NN>
    %buf3 = memref.alloc() : memref<100000xf16, @CMX_NN>
    %buf4 = memref.alloc() : memref<100000xf16, @CMX_NN>
    %buf5 = memref.alloc() : memref<100000xf16, @CMX_NN>
    %buf6 = memref.alloc() : memref<100000xf16, @CMX_NN>

    %t_dma_in, %r_dma_in = async.execute -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%in : memref<100000xf16>) outputs(%buf0 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %0 : memref<100000xf16, @CMX_NN>
    }

    // Operation that is using master buffer which will not be directly identified for spilling but for
    // which dependant operations still need to be updated as it uses spilled master buffer
    %t0, %r0 = async.execute [%t_dma_in] (%r_dma_in as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.SubView %buf_master [100000][100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
        %1 = IERT.ReLU inputs(%arg0 : memref<100000xf16, @CMX_NN>) outputs(%0 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %1 : memref<100000xf16, @CMX_NN>
    }

    // Operation that is using master buffer and will be identified as necessary for spilling
    // Dependant operations will need to be updated to refer to spillRead result
    %t1, %r1 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.SubView %buf_master [0][100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
        %1 = IERT.ReLU inputs(%arg0 : memref<100000xf16, @CMX_NN>) outputs(%0 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %1 : memref<100000xf16, @CMX_NN>
    }

    %t2, %r2 = async.execute -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%cst0 : memref<100000xf16>) outputs(%buf1 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %0 : memref<100000xf16, @CMX_NN>
    }

    %t3, %r3 = async.execute [%t1] (%r1 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.ReLU inputs(%arg0: memref<100000xf16, @CMX_NN>) outputs(%buf2 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %0 : memref<100000xf16, @CMX_NN>
    }

    %t4, %r4 = async.execute [%t3, %t2] (%r3 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>, %r2 as %arg1 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.Add inputs(%arg0: memref<100000xf16, @CMX_NN>, %arg1: memref<100000xf16, @CMX_NN>) outputs(%buf3 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %0 : memref<100000xf16, @CMX_NN>
    }

    // operation that is using buffer that will be spilled through result of async exec op
    %t5, %r5 = async.execute [%t1, %t4] (%r1 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>, %r4 as %arg1 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.Add inputs(%arg0: memref<100000xf16, @CMX_NN>, %arg1: memref<100000xf16, @CMX_NN>) outputs(%buf4 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %0 : memref<100000xf16, @CMX_NN>
    }

    // operation that is using directly master buffer that will be spilled
    %t6, %r6 = async.execute [%t5] (%r5 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.SubView %buf_master [0][100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
        %1 = IERT.Add inputs(%0: memref<100000xf16, @CMX_NN>, %arg0: memref<100000xf16, @CMX_NN>) outputs(%buf5 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %1 : memref<100000xf16, @CMX_NN>
    }

    // operation that is a user of other op that is also using master buffer which got spilled
    %t7, %r7 = async.execute [%t6] (%r0 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>, %r6 as %arg1 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16, @CMX_NN>> {
        %0 = IERT.Add inputs(%arg0: memref<100000xf16, @CMX_NN>, %arg1: memref<100000xf16, @CMX_NN>) outputs(%buf6 : memref<100000xf16, @CMX_NN>) -> memref<100000xf16, @CMX_NN>
        async.yield %0 : memref<100000xf16, @CMX_NN>
    }

    %t_tma_out, %r_dma_out = async.execute [%t7] (%r7 as %arg0 : !async.value<memref<100000xf16, @CMX_NN>>)
            -> !async.value<memref<100000xf16>> {
        %0 = IERT.Copy inputs(%arg0 : memref<100000xf16, @CMX_NN>) outputs(%out : memref<100000xf16>) -> memref<100000xf16>
        async.yield %0 : memref<100000xf16>
    }

    %result = async.await %r_dma_out : !async.value<memref<100000xf16>>
    return %result : memref<100000xf16>

    // CHECK:       [[BUF_MASTER:%.*]] = IERT.StaticAlloc<
    // CHECK-SAME:      > -> memref<200000xf16, @CMX_NN>
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<200000xf16, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = IERT.StaticAlloc<
    // CHECK-SAME:      > -> memref<200000xf16, @CMX_NN>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK:       IERT.Copy

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [100000] [100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
    // CHECK:       IERT.ReLU

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [0] [100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
    // CHECK:       IERT.ReLU

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK:       IERT.ReLU

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK:       IERT.Copy

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK:       IERT.Copy inputs([[BUF_MASTER]] : memref<200000xf16, @CMX_NN>) outputs([[BUF_SPILL_WRITE]] : memref<200000xf16, @DDR>)

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK:       IERT.Add

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-SAME:      ([[R5]] as [[ARG0:%.*]]: !async.value<memref<200000xf16, @DDR>>
    // CHECK:       IERT.Copy inputs([[ARG0]] : memref<200000xf16, @DDR>) outputs([[BUF_SPILL_READ]] : memref<200000xf16, @CMX_NN>)

    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK-SAME:      ([[R7]] as [[ARG1:%.*]]: !async.value<memref<200000xf16, @CMX_NN>>
    // CHECK:       [[SUBVIEW_0:%.*]] = IERT.SubView [[ARG1]] [0] [100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
    // CHECK:       IERT.Add inputs([[SUBVIEW_0]] : memref<100000xf16, @CMX_NN>

    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK:       [[SUBVIEW_1:%.*]] = IERT.SubView [[BUF_SPILL_READ]] [0] [100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
    // CHECK:       IERT.Add inputs([[SUBVIEW_1]] : memref<100000xf16, @CMX_NN>

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK-SAME:      ([[R7]] as [[ARG2:%.*]]: !async.value<memref<200000xf16, @CMX_NN>>
    // CHECK:       [[SUBVIEW_2:%.*]] = IERT.SubView [[ARG2]] [100000] [100000] : memref<200000xf16, @CMX_NN> to memref<100000xf16, @CMX_NN>
    // CHECK:       IERT.Add inputs([[SUBVIEW_2]] : memref<100000xf16, @CMX_NN>

}

}

// -----

// CHECK-LABEL: @SpillWriteOptimize
module @SpillWriteOptimize {

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
    %buf4 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf5 = memref.alloc() : memref<1x120000xf16, @CMX_NN>
    %buf6 = memref.alloc() : memref<1x120000xf16, @CMX_NN>

    %t_in, %r_in = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%in : memref<1x120000xf16>) outputs(%buf_in : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf0 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t1, %r1 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %1 = IERT.ReLU inputs(%0: memref<1x120000xf16, @CMX_NN>) outputs(%buf1 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %1 : memref<1x120000xf16, @CMX_NN>
    }

    %t2, %r2 = async.execute [%t0, %t1] (%r0 as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r1 as %1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, @CMX_NN>, %1: memref<1x120000xf16, @CMX_NN>) outputs(%buf2 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %2 : memref<1x120000xf16, @CMX_NN>
    }

    %t3, %r3 = async.execute [%t_in, %t2] (%r_in as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r2 as %1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, @CMX_NN>, %1: memref<1x120000xf16, @CMX_NN>) outputs(%buf3 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %2 : memref<1x120000xf16, @CMX_NN>
    }

    %t4, %r4 = async.execute [%t3] -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf4 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %0 : memref<1x120000xf16, @CMX_NN>
    }

    %t5, %r5 = async.execute [%t3, %t4] (%r3 as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r4 as %1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, @CMX_NN>, %1: memref<1x120000xf16, @CMX_NN>) outputs(%buf5 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %2 : memref<1x120000xf16, @CMX_NN>
    }

    %t6, %r6 = async.execute [%t_in, %t5] (%r_in as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>, %r5 as %1 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16, @CMX_NN>> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, @CMX_NN>, %1: memref<1x120000xf16, @CMX_NN>) outputs(%buf6 : memref<1x120000xf16, @CMX_NN>) -> memref<1x120000xf16, @CMX_NN>
        async.yield %2 : memref<1x120000xf16, @CMX_NN>
    }

    %t7, %r7 = async.execute [%t6] (%r6 as %0 : !async.value<memref<1x120000xf16, @CMX_NN>>)
            -> !async.value<memref<1x120000xf16>> {
        %1 = IERT.Copy inputs(%0 : memref<1x120000xf16, @CMX_NN>) outputs(%out : memref<1x120000xf16>) -> memref<1x120000xf16>
        async.yield %1 : memref<1x120000xf16>
    }

    %result = async.await %r7 : !async.value<memref<1x120000xf16>>
    return %result : memref<1x120000xf16>

    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x120000xf16, @DDR>
    // CHECK:       [[BUF_SPILL_READ0:%.*]] = IERT.StaticAlloc<
    // CHECK:       [[BUF_SPILL_READ1:%.*]] = IERT.StaticAlloc<

    // Operation 0 whose output will be later spilled
    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       IERT.Copy inputs(%arg0 : memref<1x120000xf16>) outputs([[BUF_TO_SPILL:%.*]] :

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       IERT.ReLU

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy

    // First SPILL WRITE for buffer from operation 0
    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy inputs([[BUF_TO_SPILL]] : memref<1x120000xf16, @CMX_NN>) outputs([[BUF_SPILL_WRITE]] : memref<1x120000xf16, @DDR>)

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Add

    // First SPILL READ of spilled buffer from operation 0
    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      ([[R3]] as [[ARG0:%.*]]: !async.value<memref<1x120000xf16, @DDR>>
    // CHECK:       IERT.Copy inputs([[ARG0]] : memref<1x120000xf16, @DDR>) outputs([[BUF_SPILL_READ0]] : memref<1x120000xf16, @CMX_NN>)

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK-SAME:      ([[R5]] as [[ARG1:%.*]]: !async.value<memref<1x120000xf16, @CMX_NN>>,
    // CHECK:       IERT.Add inputs
    // CHECK-SAME:      [[ARG1]] : memref<1x120000xf16, @CMX_NN>

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy

    // Here second SPILL WRITE of operation 0 output would be inserted if no optimization was performed
    
    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Add

    // Second SPILL READ of spilled buffer from operation 0
    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK-SAME:      ([[R3]] as [[ARG2:%.*]]: !async.value<memref<1x120000xf16, @DDR>>
    // CHECK:       IERT.Copy inputs([[ARG2]] : memref<1x120000xf16, @DDR>) outputs([[BUF_SPILL_READ1]] : memref<1x120000xf16, @CMX_NN>)

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK-SAME:      ([[R9]] as [[ARG3:%.*]]: !async.value<memref<1x120000xf16, @CMX_NN>>,
    // CHECK:       IERT.Add inputs
    // CHECK-SAME:      [[ARG3]] : memref<1x120000xf16, @CMX_NN>

    // CHECK:       [[T11:%.+]], [[R11:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy
}

}

// -----

// CHECK-LABEL: @ControlEdgeOverlapMemory
module @ControlEdgeOverlapMemory {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<200000xf16>
    }
    outputsInfo : {
        DataInfo "prob0" : tensor<200000xf16>
        DataInfo "prob1" : tensor<200000xf16>
    }
func @main(%in: memref<200000xf16>, %out0: memref<200000xf16>, %out1: memref<200000xf16>) -> (memref<200000xf16>, memref<200000xf16>) {
    %buf0 = memref.alloc() : memref<200000xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<200000xf16, @CMX_NN>
    %buf2 = memref.alloc() : memref<200000xf16, @CMX_NN>

    // Task 0
    %t0, %f0 = async.execute -> !async.value<memref<200000xf16, @CMX_NN>> {
        %0 = IERT.Copy inputs(%in : memref<200000xf16>) outputs(%buf0 : memref<200000xf16, @CMX_NN>) -> memref<200000xf16, @CMX_NN>
        async.yield %0 : memref<200000xf16, @CMX_NN>
    }

    // Task 1
    %t1, %f1 = async.execute (%f0 as %arg0 : !async.value<memref<200000xf16, @CMX_NN>>)
            -> !async.value<memref<200000xf16, @CMX_NN>> {
        %0 = IERT.ReLU inputs(%arg0: memref<200000xf16, @CMX_NN>) outputs(%buf1 : memref<200000xf16, @CMX_NN>) -> memref<200000xf16, @CMX_NN>
        async.yield %0 : memref<200000xf16, @CMX_NN>
    }

    // Task 2
    %t2, %f2 = async.execute (%f1 as %arg0 : !async.value<memref<200000xf16, @CMX_NN>>)
            -> !async.value<memref<200000xf16>> {
        %0 = IERT.Copy inputs(%arg0 : memref<200000xf16, @CMX_NN>) outputs(%out0 : memref<200000xf16>) -> memref<200000xf16>
        async.yield %0 : memref<200000xf16>
    }

    // Task 3
    %t3, %f3 = async.execute (%f0 as %arg0 : !async.value<memref<200000xf16, @CMX_NN>>)
            -> !async.value<memref<200000xf16, @CMX_NN>> {
        %0 = IERT.ReLU inputs(%arg0: memref<200000xf16, @CMX_NN>) outputs(%buf2 : memref<200000xf16, @CMX_NN>) -> memref<200000xf16, @CMX_NN>
        async.yield %0 : memref<200000xf16, @CMX_NN>
    }

    // Task 4
    %t4, %f4 = async.execute (%f3 as %arg0 : !async.value<memref<200000xf16, @CMX_NN>>)
            -> !async.value<memref<200000xf16>> {
        %0 = IERT.Copy inputs(%arg0 : memref<200000xf16, @CMX_NN>) outputs(%out1 : memref<200000xf16>) -> memref<200000xf16>
        async.yield %0 : memref<200000xf16>
    }

    %r0 = async.await %f2 : !async.value<memref<200000xf16>>
    %r1 = async.await %f4 : !async.value<memref<200000xf16>>
    return %r0, %r1 : memref<200000xf16>, memref<200000xf16>

    // Token dependencies will match data flow by default:
    //  Task0 -> Task1 -> Task2
    //  Task0 -> Task3 -> Task4
    // besides that due to overlapping memory ranges of Task3 and Task1
    // additional control edge will be inserted:
    //  Task1 -> Task3
    // Optimization of token dependencies (transitive reduction) is beyond
    // this pass and done as a separate step

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<0>
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<400000>
    // CHECK:       [[BUF2:%.*]] = IERT.StaticAlloc<400000>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:      IERT.Copy
    // CHECK-SAME:      outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK-NEXT:      IERT.ReLU
    // CHECK-SAME:      outputs([[BUF1]]

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-SAME:      [[T1]]
    // CHECK-NEXT:      IERT.Copy

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:      [[T0]], [[T1]]
    // CHECK-NEXT:      IERT.ReLU
    // CHECK-SAME:      outputs([[BUF2]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      [[T3]]
    // CHECK-NEXT:      IERT.Copy
}

}

// -----

// CHECK-LABEL: @Prefetching
module @Prefetching {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x64000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64000xf16>
    }

func @main(%in: memref<1x64000xf16>, %out: memref<1x64000xf16>) -> memref<1x64000xf16> {
    %cst0 = const.Declare memref<1x64000xf16> = #const.Content<dense<2.0> : tensor<1x64000xf16>>
    %cst1 = const.Declare memref<1x64000xf16> = #const.Content<dense<2.0> : tensor<1x64000xf16>>
    %cst2 = const.Declare memref<1x64000xf16> = #const.Content<dense<2.0> : tensor<1x64000xf16>>

    %buf_in = memref.alloc() : memref<1x64000xf16, @CMX_NN>

    %buf0 = memref.alloc() : memref<1x64000xf16, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x64000xf16, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x64000xf16, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x64000xf16, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x64000xf16, @CMX_NN>
    %buf5 = memref.alloc() : memref<1x64000xf16, @CMX_NN>

    %t_in, %r_in = async.execute -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = IERT.Copy inputs(%in : memref<1x64000xf16>) outputs(%buf_in : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %0 : memref<1x64000xf16, @CMX_NN>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = IERT.Copy inputs(%cst0 : memref<1x64000xf16>) outputs(%buf0 : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %0 : memref<1x64000xf16, @CMX_NN>
    }

    %t1, %r1 = async.execute -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = IERT.Copy inputs(%cst1 : memref<1x64000xf16>) outputs(%buf1 : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %0 : memref<1x64000xf16, @CMX_NN>
    }

    %t2, %r2 = async.execute -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = IERT.Copy inputs(%cst2 : memref<1x64000xf16>) outputs(%buf2 : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %0 : memref<1x64000xf16, @CMX_NN>
    }

    %t3, %r3 = async.execute [%t_in, %t0] (%r_in as %0 : !async.value<memref<1x64000xf16, @CMX_NN>>, %r0 as %1 : !async.value<memref<1x64000xf16, @CMX_NN>>)
            -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @NCE, IERT.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = IERT.Add inputs(%0: memref<1x64000xf16, @CMX_NN>, %1: memref<1x64000xf16, @CMX_NN>) outputs(%buf3 : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %2 : memref<1x64000xf16, @CMX_NN>
    }

    %t4, %r4 = async.execute [%t3, %t1] (%r3 as %0 : !async.value<memref<1x64000xf16, @CMX_NN>>, %r1 as %1 : !async.value<memref<1x64000xf16, @CMX_NN>>)
            -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @NCE, IERT.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %2 = IERT.Add inputs(%0: memref<1x64000xf16, @CMX_NN>, %1: memref<1x64000xf16, @CMX_NN>) outputs(%buf4 : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %2 : memref<1x64000xf16, @CMX_NN>
    }

    %t5, %r5 = async.execute [%t4, %t2] (%r4 as %0 : !async.value<memref<1x64000xf16, @CMX_NN>>, %r2 as %1 : !async.value<memref<1x64000xf16, @CMX_NN>>)
            -> !async.value<memref<1x64000xf16, @CMX_NN>> attributes {IERT.executor = @NCE, IERT.num_units = 1 : i64, "async-deps-index" = 6 : i64} {
        %2 = IERT.Add inputs(%0: memref<1x64000xf16, @CMX_NN>, %1: memref<1x64000xf16, @CMX_NN>) outputs(%buf5 : memref<1x64000xf16, @CMX_NN>) -> memref<1x64000xf16, @CMX_NN>
        async.yield %2 : memref<1x64000xf16, @CMX_NN>
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x64000xf16, @CMX_NN>>)
            -> !async.value<memref<1x64000xf16>> attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
        %1 = IERT.Copy inputs(%0 : memref<1x64000xf16, @CMX_NN>) outputs(%out : memref<1x64000xf16>) -> memref<1x64000xf16>
        async.yield %1 : memref<1x64000xf16>
    }

    %6 = async.await %r6 : !async.value<memref<1x64000xf16>>
    return %6 : memref<1x64000xf16>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       IERT.Copy inputs(%arg0 : memref<1x64000xf16>) outputs([[BUF0:%.*]] :

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Add

    // Prefetched Copy ops below

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Add

    // No stall between NCE Tasks

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Add

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-NEXT:       IERT.Copy
}

}
