// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s

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

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           MemoryResource 4096 bytes of "CMX_NN"

func @main(%in: memref<1x1000xf16>, %out: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %buf0 = memref.alloc() : memref<1x1000xf16, "CMX_NN">
    %buf1 = memref.alloc() : memref<1x1000xf16, "CMX_NN">
    %buf2 = memref.alloc() : memref<1x1000xf16, "CMX_NN">

    %t0, %f0 = async.execute -> !async.value<memref<1x1000xf16, "CMX_NN">> {
        %0 = IERT.ReLU inputs(%in : memref<1x1000xf16>) outputs(%buf0 : memref<1x1000xf16, "CMX_NN">) -> memref<1x1000xf16, "CMX_NN">
        async.yield %0 : memref<1x1000xf16, "CMX_NN">
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x1000xf16, "CMX_NN">>)
            -> !async.value<memref<1x1000xf16, "CMX_NN">> {
        %1 = IERT.ReLU inputs(%0: memref<1x1000xf16, "CMX_NN">) outputs(%buf1 : memref<1x1000xf16, "CMX_NN">) -> memref<1x1000xf16, "CMX_NN">
        async.yield %1 : memref<1x1000xf16, "CMX_NN">
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1000xf16, "CMX_NN">>)
            -> !async.value<memref<1x1000xf16, "CMX_NN">> {
        %2 = IERT.ReLU inputs(%1: memref<1x1000xf16, "CMX_NN">) outputs(%buf2 : memref<1x1000xf16, "CMX_NN">) -> memref<1x1000xf16, "CMX_NN">
        async.yield %2 : memref<1x1000xf16, "CMX_NN">
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x1000xf16, "CMX_NN">>)
            -> !async.value<memref<1x1000xf16>> {
        %3 = IERT.Copy inputs(%2 : memref<1x1000xf16, "CMX_NN">) outputs(%out : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %3 : memref<1x1000xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x1000xf16>>
    return %3 : memref<1x1000xf16>

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, "CMX_NN">
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<2048> -> memref<1x1000xf16, "CMX_NN">
    // CHECK:       [[BUF2:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, "CMX_NN">

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x1000xf16, "CMX_NN">)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x1000xf16, "CMX_NN">)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x1000xf16, "CMX_NN">)

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

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           MemoryResource 128 bytes of "CMX_NN"

func @main(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %cst = const.Declare memref<2xf16, "CMX_NN"> = #const.Content<dense<1.000000e+00> : tensor<2xf16>>
    %buf0 = memref.alloc() : memref<2xf16, "CMX_NN">
    %buf1 = memref.alloc() : memref<2xf16, "CMX_NN">
    %token, %results = async.execute -> !async.value<memref<2xf16, "CMX_NN">> attributes {"async-deps-index" = 0 : i64} {
      %4 = IERT.ReLU inputs(%arg0 : memref<2xf16>) outputs(%buf0 : memref<2xf16, "CMX_NN">) -> memref<2xf16, "CMX_NN">
      async.yield %4 : memref<2xf16, "CMX_NN">
    }
    %token_0, %results_1 = async.execute -> !async.value<memref<2xf16, "CMX_NN">> attributes {"async-deps-index" = 1 : i64} {
      %4 = IERT.ReLU inputs(%cst : memref<2xf16, "CMX_NN">) outputs(%buf1 : memref<2xf16, "CMX_NN">) -> memref<2xf16, "CMX_NN">
      async.yield %4 : memref<2xf16, "CMX_NN">
    }
    %token_2, %results_3 = async.execute [%token] (%results as %arg3: !async.value<memref<2xf16, "CMX_NN">>) -> !async.value<memref<2xf16>> attributes {"async-deps-index" = 2 : i64} {
      %4 = IERT.Copy inputs(%arg3 : memref<2xf16, "CMX_NN">) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
      async.yield %4 : memref<2xf16>
    }
    %token_4, %results_5 = async.execute [%token_0] (%results_1 as %arg3: !async.value<memref<2xf16, "CMX_NN">>) -> !async.value<memref<2xf16>> attributes {"async-deps-index" = 3 : i64} {
      %4 = IERT.Copy inputs(%arg3 : memref<2xf16, "CMX_NN">) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
      async.yield %4 : memref<2xf16>
    }
    %2 = async.await %results_3 : !async.value<memref<2xf16>>
    %3 = async.await %results_5 : !async.value<memref<2xf16>>
    return %2, %3 : memref<2xf16>, memref<2xf16>

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<0> -> memref<2xf16, "CMX_NN">
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<64> -> memref<2xf16, "CMX_NN">

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF0]] : memref<2xf16, "CMX_NN">)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF1]] : memref<2xf16, "CMX_NN">)

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

    %buf_in = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    
    %buf0 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf1 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf2 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf3 = memref.alloc() : memref<1x120000xf16, "CMX_NN">

    %t_in, %r_in = async.execute -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %0 = IERT.Copy inputs(%in : memref<1x120000xf16>) outputs(%buf_in : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %0 : memref<1x120000xf16, "CMX_NN">
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf0 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %0 : memref<1x120000xf16, "CMX_NN">
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %1 = IERT.ReLU inputs(%0: memref<1x120000xf16, "CMX_NN">) outputs(%buf1 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %1 : memref<1x120000xf16, "CMX_NN">
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>, %r0 as %1 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, "CMX_NN">, %1: memref<1x120000xf16, "CMX_NN">) outputs(%buf2 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %2 : memref<1x120000xf16, "CMX_NN">
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>, %r1 as %1 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, "CMX_NN">, %1: memref<1x120000xf16, "CMX_NN">) outputs(%buf3 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %2 : memref<1x120000xf16, "CMX_NN">
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16>> {
        %1 = IERT.Copy inputs(%0 : memref<1x120000xf16, "CMX_NN">) outputs(%out : memref<1x120000xf16>) -> memref<1x120000xf16>
        async.yield %1 : memref<1x120000xf16>
    }

    %6 = async.await %r6 : !async.value<memref<1x120000xf16>>
    return %6 : memref<1x120000xf16>

    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x120000xf16, "DDR">
    // CHECK:       [[BUF_SPILL_READ:%.*]] = IERT.StaticAlloc<
    // CHECK-SAME:      > -> memref<1x120000xf16, "CMX_NN">

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK:       IERT.Copy

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute ->
    // CHECK:       IERT.Copy

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       IERT.ReLU

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:      ([[R0:%.+]] as [[ARG0:%.*]]
    // CHECK:       IERT.Copy inputs([[ARG0:%.*]] : memref<1x120000xf16, "CMX_NN">) outputs([[BUF_SPILL_WRITE]] : memref<1x120000xf16, "DDR">)

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      ([[R3:%.+]] as [[ARG1:%.*]]
    // CHECK:       IERT.Copy inputs([[ARG1:%.*]] : memref<1x120000xf16, "DDR">) outputs([[BUF_SPILL_READ]] : memref<1x120000xf16, "CMX_NN">)

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      ([[R4:%.+]] as [[ARG2:%.*]]
    // CHECK:       IERT.Add inputs
    // CHECK-SAME:      [[ARG2:%.*]] : memref<1x120000xf16, "CMX_NN">
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
    %buf_master = memref.alloc() : memref<200000xf16, "CMX_NN">

    %buf0 = memref.alloc() : memref<100000xf16, "CMX_NN">
    %buf1 = memref.alloc() : memref<100000xf16, "CMX_NN">
    %buf2 = memref.alloc() : memref<100000xf16, "CMX_NN">
    %buf3 = memref.alloc() : memref<100000xf16, "CMX_NN">
    %buf4 = memref.alloc() : memref<100000xf16, "CMX_NN">
    %buf5 = memref.alloc() : memref<100000xf16, "CMX_NN">
    %buf6 = memref.alloc() : memref<100000xf16, "CMX_NN">

    %t_dma_in, %r_dma_in = async.execute -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.Copy inputs(%in : memref<100000xf16>) outputs(%buf0 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %0 : memref<100000xf16, "CMX_NN">
    }

    // Operation that is using master buffer which will not be directly identified for spilling but for
    // which dependant operations still need to be updated as it uses spilled master buffer
    %t0, %r0 = async.execute [%t_dma_in] (%r_dma_in as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.SubView %buf_master [100000][100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
        %1 = IERT.ReLU inputs(%arg0 : memref<100000xf16, "CMX_NN">) outputs(%0 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %1 : memref<100000xf16, "CMX_NN">
    }

    // Operation that is using master buffer and will be identified as necessary for spilling
    // Dependant operations will need to be updated to refer to spillRead result
    %t1, %r1 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.SubView %buf_master [0][100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
        %1 = IERT.ReLU inputs(%arg0 : memref<100000xf16, "CMX_NN">) outputs(%0 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %1 : memref<100000xf16, "CMX_NN">
    }

    %t2, %r2 = async.execute -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.Copy inputs(%cst0 : memref<100000xf16>) outputs(%buf1 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %0 : memref<100000xf16, "CMX_NN">
    }

    %t3, %r3 = async.execute [%t1] (%r1 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.ReLU inputs(%arg0: memref<100000xf16, "CMX_NN">) outputs(%buf2 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %0 : memref<100000xf16, "CMX_NN">
    }

    %t4, %r4 = async.execute [%t3, %t2] (%r3 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>, %r2 as %arg1 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.Add inputs(%arg0: memref<100000xf16, "CMX_NN">, %arg1: memref<100000xf16, "CMX_NN">) outputs(%buf3 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %0 : memref<100000xf16, "CMX_NN">
    }

    // operation that is using buffer that will be spilled through result of async exec op
    %t5, %r5 = async.execute [%t1, %t4] (%r1 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>, %r4 as %arg1 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.Add inputs(%arg0: memref<100000xf16, "CMX_NN">, %arg1: memref<100000xf16, "CMX_NN">) outputs(%buf4 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %0 : memref<100000xf16, "CMX_NN">
    }

    // operation that is using directly master buffer that will be spilled
    %t6, %r6 = async.execute [%t5] (%r5 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.SubView %buf_master [0][100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
        %1 = IERT.Add inputs(%0: memref<100000xf16, "CMX_NN">, %arg0: memref<100000xf16, "CMX_NN">) outputs(%buf5 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %1 : memref<100000xf16, "CMX_NN">
    }

    // operation that is a user of other op that is also using master buffer which got spilled
    %t7, %r7 = async.execute [%t6] (%r0 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>, %r6 as %arg1 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16, "CMX_NN">> {
        %0 = IERT.Add inputs(%arg0: memref<100000xf16, "CMX_NN">, %arg1: memref<100000xf16, "CMX_NN">) outputs(%buf6 : memref<100000xf16, "CMX_NN">) -> memref<100000xf16, "CMX_NN">
        async.yield %0 : memref<100000xf16, "CMX_NN">
    }

    %t_tma_out, %r_dma_out = async.execute [%t7] (%r7 as %arg0 : !async.value<memref<100000xf16, "CMX_NN">>)
            -> !async.value<memref<100000xf16>> {
        %0 = IERT.Copy inputs(%arg0 : memref<100000xf16, "CMX_NN">) outputs(%out : memref<100000xf16>) -> memref<100000xf16>
        async.yield %0 : memref<100000xf16>
    }

    %result = async.await %r_dma_out : !async.value<memref<100000xf16>>
    return %result : memref<100000xf16>

    // CHECK:       [[BUF_MASTER:%.*]] = IERT.StaticAlloc<
    // CHECK-SAME:      > -> memref<200000xf16, "CMX_NN">
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<200000xf16, "DDR">
    // CHECK:       [[BUF_SPILL_READ:%.*]] = IERT.StaticAlloc<
    // CHECK-SAME:      > -> memref<200000xf16, "CMX_NN">

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK:       IERT.Copy

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute ->
    // CHECK:       IERT.Copy

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [100000] [100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
    // CHECK:       IERT.ReLU

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [0] [100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
    // CHECK:       IERT.ReLU

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK:       IERT.ReLU

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK:       IERT.Copy inputs([[BUF_MASTER:%.*]] : memref<200000xf16, "CMX_NN">) outputs([[BUF_SPILL_WRITE]] : memref<200000xf16, "DDR">)

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK:       IERT.Add

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-SAME:      ([[R5:%.+]] as [[ARG0:%.*]]
    // CHECK:       IERT.Copy inputs([[ARG0:%.*]] : memref<200000xf16, "DDR">) outputs([[BUF_SPILL_READ]] : memref<200000xf16, "CMX_NN">)

    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK-SAME:      ([[R5:%.+]] as [[ARG1:%.*]]
    // CHECK:       [[SUBVIEW_0:%.*]] = IERT.SubView [[ARG1:%.*]] [0] [100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
    // CHECK:       IERT.Add inputs([[SUBVIEW_0:%.*]] : memref<100000xf16, "CMX_NN">

    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK:       [[SUBVIEW_1:%.*]] = IERT.SubView [[BUF_SPILL_READ]] [0] [100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
    // CHECK:       IERT.Add inputs([[SUBVIEW_1:%.*]] : memref<100000xf16, "CMX_NN">

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK-SAME:      ([[R5:%.+]] as [[ARG2:%.*]]
    // CHECK:       [[SUBVIEW_2:%.*]] = IERT.SubView [[ARG2:%.*]] [100000] [100000] : memref<200000xf16, "CMX_NN"> to memref<100000xf16, "CMX_NN">
    // CHECK:       IERT.Add inputs([[SUBVIEW_2:%.*]] : memref<100000xf16, "CMX_NN">

}

}
