// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --feasible-allocation="memory-space=CMX_NN" %s | FileCheck %s

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