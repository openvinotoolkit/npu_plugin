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
