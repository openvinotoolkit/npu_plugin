// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --assign-tensor-offsets-ddr %s | FileCheck %s

#NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @LinearGraph
module @LinearGraph {

VPUIP.Graph "LinearGraph" at @main
    options : "NONE"
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

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           IERT.MemoryResource 4096 bytes of "DDR"

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = alloc() : memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>)

    %1 = alloc() : memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32} inputs(%0 : memref<1x1000xf16>) outputs(%1 : memref<1x1000xf16>)
    dealloc %0 : memref<1x1000xf16>

    %2 = alloc() : memref<1x1000xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32} inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>)
    dealloc %1 : memref<1x1000xf16>

    VPUIP.UPADMA inputs(%2 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>)
    dealloc %2 : memref<1x1000xf16>

    return

    // CHECK-NOT:   alloc
    // CHECK:       [[VAR0:%[0-9]*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1000xf16>

    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1000xf16>)

    // CHECK-NOT:   alloc
    // CHECK:       [[VAR1:%[0-9]*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<1x1000xf16>

    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x1000xf16>)
    // CHECK-NOT:   dealloc

    // CHECK-NOT:   alloc
    // CHECK:       [[VAR2:%[0-9]*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1000xf16>

    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x1000xf16>)
    // CHECK-NOT:   dealloc

    // CHECK-NEXT:  VPUIP.UPADMA
    // CHECK-SAME:      inputs([[VAR2]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs([[ARG1]] : memref<1x1000xf16>)
    // CHECK-NOT:   dealloc

    // CHECK-NEXT:  return
}

}
