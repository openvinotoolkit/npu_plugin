// RUN: vpux-opt -split-input-file -assign-tensor-offsets-ddr %s | FileCheck %s

#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: ddr_scratch = 4096
VPUIP.Graph "LinearGraph" at @main
    options : "NONE"
    resources : {
        nn_cmx_slice_amount = 1 : i32,
        upa_shaves = 1 : i32
    }
    inputsInfo : {
        VPUIP.TensorInfo "data", f16, #NC
    }
    outputsInfo : {
        VPUIP.TensorInfo "prob", f16, #NC
    }

func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    // CHECK: "VPU_DDR_Heap", 0
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)

    // CHECK: "VPU_DDR_Heap", 2048
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%0 : memref<1x1000xf16>) outputs(%1 : memref<1x1000xf16>)

    // CHECK: "VPU_DDR_Heap", 0
    %2 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>)

    VPUIP.UPADMA inputs(%2 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)

    return
}
