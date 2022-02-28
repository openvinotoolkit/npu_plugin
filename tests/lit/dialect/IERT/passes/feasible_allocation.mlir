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

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF0]] : memref<2xf16, @CMX_NN>)

    // CHECK:       IERT.Copy

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
    //  Task2 -> Task3
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
    // CHECK-SAME:      [[T0]], [[T2]]
    // CHECK-NEXT:      IERT.ReLU
    // CHECK-SAME:      outputs([[BUF2]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      [[T3]]
    // CHECK-NEXT:      IERT.Copy
}

}

// -----

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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


!InputDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    num_clusters = 4
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type memref<1x32x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<64x32x3x3xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<64x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = type memref<1x64x16x16xf16, #NHWC, @DDR>

!WeightsTableStub = type memref<64x1x1x4xsi32>
!InputStub_CMX = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<64x1x1x4xsi32, @CMX_NN>
!OutputStub_CMX = type memref<1x64x16x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SingleConvWithClustering
module @SingleConvWithClustering {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x16x16xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64x16x16xf16>
    }

func @main(%input: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = #const.Content<dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %t0 = async.execute
            attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
            %1 = IERT.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
        }

        async.yield
    }

    %t1 = async.execute
            attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
            %1 = IERT.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
        }

        async.yield
    }

    %t2 = async.execute
            attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%weights_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
            %1 = IERT.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
        }

        async.yield
    }

    %t3 = async.execute [%t0, %t1, %t2]
                attributes {IERT.executor = @NCE, IERT.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
            %0 = VPUIP.NCEClusterTiling
                    inputs(%input_cmx as %arg0: !InputStub_CMX,
                            %weights_cmx as %arg1: !WeightsStub_CMX,
                            %weights_table_cmx as %arg2: !WeightsTableStub_CMX)
                    outputs(%output_buff_cmx as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

                  %1 = VPUIP.NCEClusterTask {
                            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = "CONV"
                        }  input(%arg0 : !InputStub_CMX)
                            weights(%arg1 : !WeightsStub_CMX)
                            weight_table(%arg2 : !WeightsTableStub_CMX)
                            parent_input(%arg0 : !InputStub_CMX)
                            parent_output(%arg3 : !OutputStub_CMX)
                            outputs(%arg3 : !OutputStub_CMX)
                                -> !OutputStub_CMX variants :  {
                            DPUTask {
                                start = [0, 0, 0], end = [31, 15, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
                            }
                            } PPE :  {
                            }
            }

            async.yield
    }

    %t4 = async.execute [%t3]
            attributes {IERT.executor = @DMA_NN, IERT.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%output_buff_cmx as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
            %1 = IERT.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
        }

        async.yield
    }

    return %output: !Output_DDR


    // CHECK:       [[CST_WEIGHTS:%.*]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[CST_WEIGHTS_TABLE:%.*]] = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR>
    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <4096> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <49152> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF3:%.*]] = VPURT.DeclareBuffer "CMX_NN" <40960> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF4:%.*]] = memref.alloc() : memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[T0:%.*]] = async.execute
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0 as [[ARG0:%.*]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF0]] as [[ARG1:%.*]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG0]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:                  outputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)   

    // CHECK:       [[T1:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CST_WEIGHTS]] as [[ARG2:%.*]]: memref<64x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF1]] as [[ARG3:%.*]]: memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG2]] : memref<64x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:                  outputs([[ARG3]] : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[T2:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CST_WEIGHTS_TABLE]] as [[ARG4:%.*]]: memref<64x1x1x4xsi32, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF2]] as [[ARG5:%.*]]: memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG4]] : memref<64x1x1x4xsi32, #NHWC, @DDR>)
    // CHECK-SAME:                  outputs([[ARG5]] : memref<64x1x1x4xsi32, @CMX_NN>)

    // CHECK:       [[T3:%.*]] = async.execute
    // CHECK-SAME:      [[T0]], [[T1]], [[T2]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[BUF0]] as [[ARG6:%.*]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, 
    // CHECK-SAME:                 [[BUF1]] as [[ARG7:%.*]]: memref<64x32x3x3xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                 [[BUF2]] as [[ARG8:%.*]]: memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUF3]] as [[ARG9:%.*]]: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:                   VPUIP.NCEClusterTask
    // CHECK-SAME:                  input([[ARG6]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                  weights([[ARG7]] : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                  weight_table([[ARG8]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:                  outputs([[ARG9]] : memref<1x64x16x16xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[T4:%.*]] = async.execute
    // CHECK-SAME:      [[T1]], [[T2]], [[T3]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[BUF3]] as [[ARG10:%.*]]: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUF4]] as [[ARG11:%.*]]: memref<1x64x16x16xf16, #NHWC, @DDR>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG10]] : memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                  outputs([[ARG11]] : memref<1x64x16x16xf16, #NHWC, @DDR>)

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufDistributed = type !VPUIP.DistributedBuffer<
    1x1x1x120000xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!BufMemrefDDR = type memref<1x1x1x120000xf16, #NHWC, @DDR>
!BufMemrefCMX = type memref<1x1x1x120000xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SpillingWithClustering
module @SpillingWithClustering {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x120000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x120000xf16>
    }

func @main(%input: !BufMemrefDDR) -> !BufMemrefDDR {
    %cst0 = const.Declare memref<1x1x1x120000xf16> = #const.Content<dense<2.0> : tensor<1x1x1x120000xf16>>

    %buf_in = VPURT.AllocDistributed -> !BufDistributed
    %buf0 = VPURT.AllocDistributed -> !BufDistributed
    %buf1 = VPURT.AllocDistributed -> !BufDistributed
    %buf2 = VPURT.AllocDistributed -> !BufDistributed
    %buf3 = VPURT.AllocDistributed -> !BufDistributed
    %output = memref.alloc() : !BufMemrefDDR

    %t_in, %r_in = async.execute -> !async.value<!BufDistributed> {
        %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !BufMemrefDDR) outputs(%buf_in as %arg1: !BufMemrefCMX) -> !BufDistributed {
            %1 = IERT.Copy inputs(%arg0 : !BufMemrefDDR) outputs(%arg1 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t0, %r0 = async.execute -> !async.value<!BufDistributed> {
        %0 = VPUIP.NCEClusterTiling inputs(%cst0 as %arg0: !BufMemrefDDR) outputs(%buf0 as %arg1: !BufMemrefCMX) -> !BufDistributed {
            %1 = IERT.Copy inputs(%arg0 : !BufMemrefDDR) outputs(%arg1 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %async_arg0 : !async.value<!BufDistributed>) -> !async.value<!BufDistributed> {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX) outputs(%buf1 as %arg1: !BufMemrefCMX) -> !BufDistributed {
            %1 = IERT.ReLU inputs(%arg0: !BufMemrefCMX) outputs(%arg1 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t1, %r1 = async.execute [%t0, %t3] (%r0 as %async_arg0 : !async.value<!BufDistributed>, %r3 as %async_arg1 : !async.value<!BufDistributed>) -> !async.value<!BufDistributed> {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX, %async_arg1 as %arg1: !BufMemrefCMX) outputs(%buf2 as %arg2: !BufMemrefCMX) -> !BufDistributed {
            %1 = IERT.Add inputs(%arg0: !BufMemrefCMX, %arg1: !BufMemrefCMX) outputs(%arg2 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %async_arg0 : !async.value<!BufDistributed>, %r1 as %async_arg1 : !async.value<!BufDistributed>) -> !async.value<!BufDistributed> {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX, %async_arg1 as %arg1: !BufMemrefCMX) outputs(%buf3 as %arg2: !BufMemrefCMX) -> !BufDistributed {
            %1 = IERT.Add inputs(%arg0: !BufMemrefCMX, %arg1: !BufMemrefCMX) outputs(%arg2 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %async_arg0 : !async.value<!BufDistributed>) -> !async.value<!BufMemrefDDR> {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX) outputs(%output as %arg1: !BufMemrefDDR) -> !BufMemrefDDR {
            %1 = IERT.Copy { out_mem_space = @DDR } inputs(%arg0: !BufMemrefCMX) outputs(%arg1: !BufMemrefDDR) -> !BufMemrefDDR
        }
        async.yield %0: !BufMemrefDDR
    }

     %6 = async.await %r6 : !async.value<!BufMemrefDDR>
     return %6 : !BufMemrefDDR

    // CHECK:       [[CST:%.*]] = const.Declare
    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <480000> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <240000> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF3:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF4:%.*]] = VPURT.DeclareBuffer "CMX_NN" <480000> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF5:%.*]] = memref.alloc() : memref<1x1x1x120000xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x1x1x120000xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = VPURT.DeclareBuffer "CMX_NN" <240000> -> !VPUIP.DistributedBuffer

    // CHECK:       [[T0:%.*]], [[R0:%.*]] = async.execute
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0 as [[ARG0:%.*]]: memref<1x1x1x120000xf16, #NHWC, @DDR>) outputs([[BUF0]] as [[ARG1:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG0]] :
    // CHECK-SAME:                  outputs([[ARG1]] :

    // CHECK:       [[T1:%.*]], [[R1:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK-SAME:      ([[R0]] as [[ASYNC_ARG0:%.*]]: !async.value<!VPUIP.DistributedBuffer
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ASYNC_ARG0]] as [[ARG3:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) outputs([[BUF2]] as [[ARG4:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.ReLU
    // CHECK-SAME:                  inputs([[ARG3]] :
    // CHECK-SAME:                  outputs([[ARG4]] : 

    // CHECK:       [[T2:%.*]], [[R2:%.*]] = async.execute
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CST]] as [[ARG5:%.*]]: memref<1x1x1x120000xf16, #NHWC, @DDR>) outputs([[BUF1]] as [[ARG6:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG5]] :
    // CHECK-SAME:                  outputs([[ARG6]] :

    // CHECK:       [[T3:%.*]], [[R3:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[BUF0]] as [[ARG7:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) outputs([[BUF_SPILL_WRITE]] as [[ARG8:%.*]]: memref<1x1x1x120000xf16, #NHWC, @DDR>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG7]] :
    // CHECK-SAME:                  outputs([[ARG8]] :

    // CHECK:       [[T4:%.*]], [[R4:%.*]] = async.execute
    // CHECK-SAME:      [[T2]], [[T1]], [[T3]]
    // CHECK-SAME:      ([[R2]] as [[ASYNC_ARG1:%.*]]: !async.value<!VPUIP.DistributedBuffer<1x1x1x120000xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}>>, [[R1]] as [[ASYNC_ARG2:%.*]]: !async.value<!VPUIP.DistributedBuffer<1x1x1x120000xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}>>)
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ASYNC_ARG1]] as [[ARG9:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>, [[ASYNC_ARG2]] as [[ARG10:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) outputs([[BUF3]] as [[ARG11:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Add
    // CHECK-SAME:                  inputs([[ARG9]] : memref<1x1x1x120000xf16, #NHWC, @CMX_NN>, [[ARG10]] : memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                  outputs([[ARG11]] : memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) 

    // CHECK:       [[T5:%.*]], [[R5:%.*]] = async.execute
    // CHECK-SAME:      [[T4]], [[T3]]
    // CHECK-SAME:      ([[R3]] as [[ASYNC_ARG3:%.*]]: !async.value<memref<1x1x1x120000xf16, #NHWC, @DDR>>
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ASYNC_ARG3]] as [[ARG12:%.*]]: memref<1x1x1x120000xf16, #NHWC, @DDR>) outputs([[BUF_SPILL_READ]] as [[ARG13:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG12]] :
    // CHECK-SAME:                  outputs([[ARG13]] :

    // CHECK:       [[T6:%.*]], [[R6:%.*]] = async.execute
    // CHECK-SAME:      [[T0]], [[T4]], [[T5]]
    // CHECK-SAME:      ([[R5]] as [[ASYNC_ARG4:%.*]]: !async.value<!VPUIP.DistributedBuffer<1x1x1x120000xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}>>, [[R4]] as [[ASYNC_ARG5:%.*]]: !async.value<!VPUIP.DistributedBuffer<1x1x1x120000xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}>>)
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ASYNC_ARG4]] as [[ARG14:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>, [[ASYNC_ARG5]] as [[ARG15:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) outputs([[BUF4]] as [[ARG16:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK:                   IERT.Add
    // CHECK-SAME:                  inputs([[ARG14]] : memref<1x1x1x120000xf16, #NHWC, @CMX_NN>, [[ARG15]] : memref<1x1x1x120000xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                  outputs([[ARG16]] : memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) 

    // CHECK:       [[T7:%.*]], [[R7:%.*]] = async.execute
    // CHECK-SAME:      [[T6]]
    // CHECK-SAME:      ([[R6]] as [[ASYNC_ARG6:%.*]]: !async.value<!VPUIP.DistributedBuffer
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[ASYNC_ARG6]] as [[ARG17:%.*]]: memref<1x1x1x120000xf16, #NHWC, @CMX_NN>) outputs([[BUF5]] as [[ARG18:%.*]]: memref<1x1x1x120000xf16, #NHWC, @DDR>)
    // CHECK:                   IERT.Copy
    // CHECK-SAME:                  inputs([[ARG17]] :
    // CHECK-SAME:                  outputs([[ARG18]] :
}

}
