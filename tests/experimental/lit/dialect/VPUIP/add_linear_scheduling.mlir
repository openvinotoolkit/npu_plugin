// RUN: vpux-opt -split-input-file -add-linear-scheduling %s | FileCheck %s

// CHECK-LABEL: linear_dma_graph

#map = affine_map<(d0, d1) -> (d0, d1)>

VPUIP.Graph {
    entryPoint = @main, identifier = "linear_dma_graph",
    // CHECK: options = [#VPUIP<"ExecutionFlag:DynamicBarriers">]
    options = [],
    resources = {
        nn_cmx_slice_amount = 1 : i32,
        upa_shaves = 1 : i32
    }
} inputsInfo {
    VPUIP.TensorInfo {layout = #map, name = "data", precision = f16}
} outputsInfo {
    VPUIP.TensorInfo {layout = #map, name = "prob", precision = f16}
}

func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    VPUIP.UPADMA inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)
    // CHECK:       %[[B0:.*]] = VPUIP.ConfigureBarrier
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      updates(%[[B0]] : !VPUIP.Barrier)

    %1 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    VPUIP.UPADMA inputs(%0 : memref<1x1000xf16>) outputs(%1 : memref<1x1000xf16>)
    // CHECK:       %[[B1:.*]] = VPUIP.ConfigureBarrier
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      waits(%[[B0]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates(%[[B1]] : !VPUIP.Barrier)

    %2 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    VPUIP.UPADMA inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>)
    // CHECK:       %[[B2:.*]] = VPUIP.ConfigureBarrier
    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      waits(%[[B1]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates(%[[B2]] : !VPUIP.Barrier)

    VPUIP.UPADMA inputs(%2 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)
    // CHECK:       VPUIP.UPADMA
    // CHECK-SAME:      waits(%[[B2]] : !VPUIP.Barrier)

    return
}

// -----

// CHECK-LABEL: linear_upa_graph

#map = affine_map<(d0, d1) -> (d0, d1)>

VPUIP.Graph {
    entryPoint = @main, identifier = "linear_upa_graph",
    // CHECK: options = [#VPUIP<"ExecutionFlag:DynamicBarriers">]
    options = [],
    resources = {
        nn_cmx_slice_amount = 1 : i32,
        upa_shaves = 1 : i32
    }
} inputsInfo {
    VPUIP.TensorInfo {layout = #map, name = "data", precision = f16}
} outputsInfo {
    VPUIP.TensorInfo {layout = #map, name = "prob", precision = f16}
}

func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)
    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      isTrailingSWLayer

    %1 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%0 : memref<1x1000xf16>) outputs(%1 : memref<1x1000xf16>)
    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      isTrailingSWLayer

    %2 = VPUIP.DeclareTensor {location = #VPUIP<"MemoryLocation:VPU_DDR_Heap">} -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%1 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)
    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      isTrailingSWLayer

    return
}
