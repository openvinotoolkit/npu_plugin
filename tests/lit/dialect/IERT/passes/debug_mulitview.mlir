// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s

module @SimpleGraph {

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

    %buf0 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf1 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf2 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf3 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf4 = memref.alloc() : memref<1x120000xf16, "CMX_NN">
    %buf5 = memref.alloc() : memref<1x120000xf16, "CMX_NN">

    %t0, %f0 = async.execute -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %0 = IERT.Copy inputs(%in : memref<1x120000xf16>) outputs(%buf0 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %0 : memref<1x120000xf16, "CMX_NN">
    }

    %t1, %f1:2 = async.execute [%t0] (%f0 as %arg0 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> (!async.value<memref<1x120000xf16, "CMX_NN">>, !async.value<memref<1x120000xf16, "CMX_NN">>) {
        %1 = IERT.ReLU inputs(%arg0: memref<1x120000xf16, "CMX_NN">) outputs(%buf2 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        %2 = IERT.ReLU inputs(%arg0: memref<1x120000xf16, "CMX_NN">) outputs(%buf1 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %1, %2 : memref<1x120000xf16, "CMX_NN">, memref<1x120000xf16, "CMX_NN">
    }

    %t_cst, %r_cst = async.execute -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %0 = IERT.Copy inputs(%cst0 : memref<1x120000xf16>) outputs(%buf3 : memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %0 : memref<1x120000xf16, "CMX_NN">
    }

    %t4, %f4 = async.execute [%t1, %t_cst] (%f1#0 as %arg0 : !async.value<memref<1x120000xf16, "CMX_NN">>, %r_cst as %arg1 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %2 = IERT.Add inputs(%arg0: memref<1x120000xf16, "CMX_NN">, %arg1: memref<1x120000xf16, "CMX_NN">) outputs(%buf4: memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %2 : memref<1x120000xf16, "CMX_NN">
    }

    %t2, %f2 = async.execute [%t1, %t4] (%f1#1 as %arg0 : !async.value<memref<1x120000xf16, "CMX_NN">>, %f4 as %arg1 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16, "CMX_NN">> {
        %2 = IERT.Add inputs(%arg0: memref<1x120000xf16, "CMX_NN">, %arg1: memref<1x120000xf16, "CMX_NN">) outputs(%buf5: memref<1x120000xf16, "CMX_NN">) -> memref<1x120000xf16, "CMX_NN">
        async.yield %2 : memref<1x120000xf16, "CMX_NN">
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x120000xf16, "CMX_NN">>)
            -> !async.value<memref<1x120000xf16>> {
        %3 = IERT.Copy inputs(%2 : memref<1x120000xf16, "CMX_NN">) outputs(%out : memref<1x120000xf16>) -> memref<1x120000xf16>
        async.yield %3 : memref<1x120000xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x120000xf16>>
    return %3 : memref<1x120000xf16>
}

}