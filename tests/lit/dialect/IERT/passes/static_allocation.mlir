// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --static-allocation="memory-space=DDR" %s | FileCheck %s

// CHECK-LABEL: @LinearGraph
module @LinearGraph {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1000xf16>
    }

// CHECK:   module @UsedMemory
// CHECK:           IE.MemoryResource 4096 bytes of @DDR

func @main(%in: memref<1x1000xf16>, %out: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %buf0 = memref.alloc() : memref<1x1000xf16, @DDR>
    %buf1 = memref.alloc() : memref<1x1000xf16, @DDR>
    %buf2 = memref.alloc() : memref<1x1000xf16, @DDR>

    %t0, %f0 = async.execute -> !async.value<memref<1x1000xf16, @DDR>> {
        %0 = IERT.ReLU inputs(%in : memref<1x1000xf16>) outputs(%buf0 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        async.yield %0 : memref<1x1000xf16, @DDR>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x1000xf16, @DDR>>)
            -> !async.value<memref<1x1000xf16, @DDR>> {
        %1 = IERT.ReLU inputs(%0: memref<1x1000xf16, @DDR>) outputs(%buf1 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        async.yield %1 : memref<1x1000xf16, @DDR>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1000xf16, @DDR>>)
            -> !async.value<memref<1x1000xf16, @DDR>> {
        %2 = IERT.ReLU inputs(%1: memref<1x1000xf16, @DDR>) outputs(%buf2 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        async.yield %2 : memref<1x1000xf16, @DDR>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x1000xf16, @DDR>>)
            -> !async.value<memref<1x1000xf16>> {
        %3 = IERT.Copy inputs(%2 : memref<1x1000xf16, @DDR>) outputs(%out : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %3 : memref<1x1000xf16>
    }

    %3 = async.await %f3 : !async.value<memref<1x1000xf16>>
    return %3 : memref<1x1000xf16>

    // CHECK:       [[BUF0:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, @DDR>
    // CHECK:       [[BUF1:%.*]] = IERT.StaticAlloc<2048> -> memref<1x1000xf16, @DDR>
    // CHECK:       [[BUF2:%.*]] = IERT.StaticAlloc<0> -> memref<1x1000xf16, @DDR>

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x1000xf16, @DDR>)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x1000xf16, @DDR>)

    // CHECK:       IERT.ReLU
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x1000xf16, @DDR>)

    // CHECK:       IERT.Copy
}

}
