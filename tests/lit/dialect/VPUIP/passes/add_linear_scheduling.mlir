// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --add-linear-scheduling %s | FileCheck %s

// CHECK-LABEL: @linear_dma_graph
module @linear_dma_graph {

VPUIP.Graph
    // CHECK: options : "NONE"
    options : "NONE"
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32, hash = "",
        contextStr = "VPUX Compiler"
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1x1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1x1x1000xf16>
    }

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16>
    VPUIP.UPADMA inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16>)
    // CHECK:       %[[B0:.*]] = VPUIP.ConfigureBarrier<0>
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      updates(%[[B0]] : !VPUIP.Barrier)

    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<1x1x1x1000xf16>
    VPUIP.UPADMA inputs(%0 : memref<1x1x1x1000xf16>) outputs(%1 : memref<1x1x1x1000xf16>)
    // CHECK:       %[[B1:.*]] = VPUIP.ConfigureBarrier<1>
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      waits(%[[B0]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates(%[[B1]] : !VPUIP.Barrier)

    %2 = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16>
    VPUIP.UPADMA inputs(%1 : memref<1x1x1x1000xf16>) outputs(%2 : memref<1x1x1x1000xf16>)
    // CHECK:       %[[B2:.*]] = VPUIP.ConfigureBarrier<0>
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      waits(%[[B1]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates(%[[B2]] : !VPUIP.Barrier)

    VPUIP.UPADMA inputs(%2 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>)
    // CHECK:       VPUIP.UPADMA
    // CHECK-SAME:      waits(%[[B2]] : !VPUIP.Barrier)

    return %arg1 : memref<1x1x1x1000xf16>
}

}

// -----

// CHECK-LABEL: @linear_upa_graph
module @linear_upa_graph {

VPUIP.Graph
    // CHECK: options : "NONE"
    options : "NONE"
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32, hash = "",
        contextStr = "VPUX Compiler"
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1x1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1x1x1000xf16>
    }

func @main(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 3 : i32} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16>)
    // CHECK:           VPUIP.SoftMaxUPA
    // CHECK-SAME:      updates(%1 : !VPUIP.Barrier)

    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<1x1x1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 3 : i32} inputs(%0 : memref<1x1x1x1000xf16>) outputs(%1 : memref<1x1x1x1000xf16>)
    // CHECK:           VPUIP.SoftMaxUPA
    // CHECK-SAME:      waits(%1 : !VPUIP.Barrier) updates(%3 : !VPUIP.Barrier)

    %2 = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1x1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 3 : i32} inputs(%1 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>)
    // CHECK:           VPUIP.SoftMaxUPA
    // CHECK-SAME:      waits(%3 : !VPUIP.Barrier)

    return  %arg1 : memref<1x1x1x1000xf16>
}

}
