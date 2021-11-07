// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s

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

    %buf3 = memref.alloc() : memref<1x120000xf16, "CMX_NN">

    %buf5 = memref.alloc() : memref<1x120000xf16, "CMX_NN">

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
        %1 = IERT.ReLU inputs(%0: memref<1x120000xf16, "CMX_NN">) outputs(%buf3 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %1 : memref<1x120000xf16, "CMX_NN">
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>, %r0 as %1 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, "CMX_NN">, %1: memref<1x120000xf16, "CMX_NN">) outputs(%buf1 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %2 : memref<1x120000xf16, "CMX_NN">
    }


    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>, %r1 as %1 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %2 = IERT.Add inputs(%0: memref<1x120000xf16, "CMX_NN">, %1: memref<1x120000xf16, "CMX_NN">) outputs(%buf5 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %2 : memref<1x120000xf16, "CMX_NN">
    }


    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16>> {
        %1 = IERT.Copy inputs(%0 : memref<1x120000xf16, "CMX_NN">) outputs(%out : memref<1x120000xf16>) -> memref<1x120000xf16>
        async.yield %1 : memref<1x120000xf16>
    }

    %6 = async.await %r6 : !async.value<memref<1x120000xf16>>
    return %6 : memref<1x120000xf16>

    // CHECK:       IERT.ReLU

    // CHECK:       IERT.Copy
}

}
