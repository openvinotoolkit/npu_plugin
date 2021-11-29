// RUN: vpux-opt --init-compiler="vpu-arch=KMB" --assign-physical-barriers %s | FileCheck %s

func @main(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<0>
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer "VPU_DDR_Heap" <0> -> memref<10xf16>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) op :  {
        %0 = VPUIP.NNDMA
            inputs(
                %arg0 : memref<10xf16>
            ) outputs(
                %buf0 : memref<10xf16>
            ) -> memref<10xf16>
    }
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<1>
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf1 = VPURT.DeclareBuffer "VPU_DDR_Heap" <2048> -> memref<10xf16>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) op :  {
        %1 = VPUIP.NNDMA
            inputs(
                %buf0 : memref<10xf16>
            ) outputs(
                %buf1 : memref<10xf16>
            ) -> memref<10xf16>
    }
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<2>
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) op :  {
        %2 = VPUIP.NNDMA
            inputs(
                %buf1 : memref<10xf16>
            ) outputs(
                %arg1 : memref<10xf16>
            ) -> memref<10xf16>
    }
    return %arg1 : memref<10xf16>
}
