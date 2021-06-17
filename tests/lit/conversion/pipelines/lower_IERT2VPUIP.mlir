// RUN: vpux-opt --lower-IERT-to-VPUIP %s | FileCheck %s

func @main(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = IERT.StaticAlloc<0> -> memref<10xf16, "DDR">
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, "DDR">>
            attributes { IERT.executor = "DMA_NN", IERT.num_units = 1 } {
        %0 = IERT.Copy inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, "DDR">) -> memref<10xf16, "DDR">
        async.yield %0 : memref<10xf16, "DDR">
    }

    %t1, %f1 = async.execute[%t0] (%f0 as %0: !async.value<memref<10xf16, "DDR">>) -> !async.value<memref<10xf16>>
            attributes { IERT.executor = "DMA_NN", IERT.num_units = 1 } {
        %1 = IERT.Copy inputs(%0 : memref<10xf16, "DDR">) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    %1 = async.await %f1 : !async.value<memref<10xf16>>
    return %1 : memref<10xf16>

    // CHECK:       [[BUF0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
    // CHECK:       [[B0:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier
    // CHECK:       [[VAL0:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, "DDR">
    // CHECK-SAME:      updates([[B0]] : !VPUIP.Barrier)

    // CHECK:       [[VAL1:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAL0]] : memref<10xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<10xf16>)
    // CHECK-SAME:      waits([[B0]] : !VPUIP.Barrier)

    // CHECK:       return [[VAL1]] : memref<10xf16>
}
