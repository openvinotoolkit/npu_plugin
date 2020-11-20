// RUN: vpux-opt -split-input-file -remove-extra-dma %s | FileCheck %s

// CHECK-LABEL: copy_block_args

func @copy_block_args(%arg0:  memref<8x8xf16>, %arg1: memref<8x8xf16>) {
    VPUIP.UPADMA inputs(%arg0 : memref<8x8xf16>) outputs(%arg1 : memref<8x8xf16>)
    // CHECK: VPUIP.UPADMA inputs(%arg0 : memref<8x8xf16>) outputs(%arg1 : memref<8x8xf16>)

    return
}

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: different_types

func @different_types() {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16>
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16, #map>
    %2 = VPUIP.DeclareTensor "VPU_CMX_UPA" -> memref<8x8xf16, 2>

    VPUIP.UPADMA inputs(%0 : memref<8x8xf16>) outputs(%1 : memref<8x8xf16, #map>)
    // CHECK: VPUIP.UPADMA inputs(%0 : memref<8x8xf16>) outputs(%1 : memref<8x8xf16, #map>)

    VPUIP.UPADMA inputs(%0 : memref<8x8xf16>) outputs(%2 : memref<8x8xf16, 2>)
    // CHECK: VPUIP.UPADMA inputs(%0 : memref<8x8xf16>) outputs(%2 : memref<8x8xf16, 2>)

    return
}

// -----

// CHECK-LABEL: copy_to_argument

func @copy_to_argument(%arg0:  memref<8x8xf16>, %arg1:  memref<8x8xf16>) {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<8x8xf16>) outputs(%0 : memref<8x8xf16>)
    VPUIP.UPADMA inputs(%0 : memref<8x8xf16>) outputs(%arg1 : memref<8x8xf16>)
    return

    // CHECK:       VPUIP.SoftMaxUPA {{.*}} inputs(%arg0 : memref<8x8xf16>) outputs(%arg1 : memref<8x8xf16>)
    // CHECK-NEXT:  return
}

// -----

// CHECK-LABEL: redirect_dst_users

func @redirect_dst_users(%arg0:  memref<8x8xf16>, %arg1:  memref<8x8xf16>) {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16>
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%arg0 : memref<8x8xf16>) outputs(%0 : memref<8x8xf16>)
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16>
    VPUIP.UPADMA inputs(%0 : memref<8x8xf16>) outputs(%1 : memref<8x8xf16>)
    VPUIP.SoftMaxUPA {axisInd = 1 : i32, maxShaves = 1 : i32} inputs(%1 : memref<8x8xf16>) outputs(%arg1 : memref<8x8xf16>)
    return

    // CHECK:       %0 = VPUIP.DeclareTensor
    // CHECK-NEXT:  VPUIP.SoftMaxUPA {{.*}} inputs(%arg0 : memref<8x8xf16>) outputs(%0 : memref<8x8xf16>)
    // CHECK-NEXT:  VPUIP.SoftMaxUPA {{.*}} inputs(%0 : memref<8x8xf16>) outputs(%arg1 : memref<8x8xf16>)
    // CHECK-NEXT:  return
}
