// RUN: vpux-opt -split-input-file -assign-tensor-offsets-ddr %s | FileCheck %s

#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: {item = "DDR", number = 4096 : i64}
VPUIP.Graph "LinearGraph" at @main
    options : "NONE"
    resources : {
        processor_allocation = [
            {item = "SHAVE_UPA", number = 1 : i64},
            {item = "NCE_Cluster", number = 1 : i64}
        ],
        processor_frequencies = [],
        memory_sizes = [
            {item = "CMX_NN", number = 1048576 : i64}
        ],
        memory_bandwidth = []
    }
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32, hash = "",
        contextStr = "VPUX Compiler"
    }
    inputsInfo : {
        VPUIP.TensorInfo "data", f16, #NC
    }
    outputsInfo : {
        VPUIP.TensorInfo "prob", f16, #NC
    }

func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    // CHECK: dataIndex = 0
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)

    // CHECK: dataIndex = 2048
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%0 : memref<1x1000xf16>) outputs(%1 : memref<1x1000xf16>)

    // CHECK: dataIndex = 0
    %2 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>)

    VPUIP.UPADMA inputs(%2 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)

    return
}
