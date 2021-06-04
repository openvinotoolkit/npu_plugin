// RUN: vpux-opt --set-compile-params="vpu-arch=VPU3700" --assign-physical-barriers %s | FileCheck %s

func @main(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    // CHECK-NOT: VPUIP.DeclareVirtualBarrier
    // CHECK: VPUIP.ConfigureBarrier<0>
    %bar0 = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier
    %buf0 = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<10xf16>
    %0 = VPUIP.NNDMA
        inputs(
            %arg0 : memref<10xf16>
        ) outputs(
            %buf0 : memref<10xf16>
        ) updates(
            %bar0 : !VPUIP.Barrier
        ) -> memref<10xf16>

    // CHECK-NOT: VPUIP.DeclareVirtualBarrier
    // CHECK: VPUIP.ConfigureBarrier<1>
    %bar1 = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier
    %buf1 = VPUIP.DeclareTensor "VPU_DDR_Heap" <2048> -> memref<10xf16>
    %1 = VPUIP.NNDMA
        inputs(
            %0 : memref<10xf16>
        ) outputs(
            %buf1 : memref<10xf16>
        ) waits(
            %bar0 : !VPUIP.Barrier
        ) updates(
            %bar1 : !VPUIP.Barrier
        ) -> memref<10xf16>

    // CHECK-NOT: VPUIP.DeclareVirtualBarrier
    // CHECK: VPUIP.ConfigureBarrier<2>
    %bar2 = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier
    %2 = VPUIP.NNDMA
        inputs(
            %1 : memref<10xf16>
        ) outputs(
            %arg1 : memref<10xf16>
        ) waits(
            %bar1 : !VPUIP.Barrier
        ) updates(
            %bar2 : !VPUIP.Barrier
        ) -> memref<10xf16>

    return %2 : memref<10xf16>
}
