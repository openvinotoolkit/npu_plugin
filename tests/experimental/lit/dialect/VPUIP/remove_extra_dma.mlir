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
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16, "DDR">
    %1 = VPUIP.DeclareTensor "VPU_DDR_Heap" -> memref<8x8xf16, #map, "DDR">
    %2 = VPUIP.DeclareTensor "VPU_CMX_UPA" -> memref<8x8xf16, "CMX_UPA">

    VPUIP.UPADMA inputs(%0 : memref<8x8xf16, "DDR">) outputs(%1 : memref<8x8xf16, #map, "DDR">)
    // CHECK: VPUIP.UPADMA inputs(%0 : memref<8x8xf16, "DDR">) outputs(%1 : memref<8x8xf16, #map, "DDR">)

    VPUIP.UPADMA inputs(%0 : memref<8x8xf16, "DDR">) outputs(%2 : memref<8x8xf16, "CMX_UPA">)
    // CHECK: VPUIP.UPADMA inputs(%0 : memref<8x8xf16, "DDR">) outputs(%2 : memref<8x8xf16, "CMX_UPA">)

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

// -----

// CHECK-LABEL: constant_op

func @constant_op(%arg0: memref<1x2x2x2xf16>) {
    %0 = VPUIP.DeclareConstantTensorOp
        dense<[
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0]
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0]
                ]
            ]
        ]> : tensor<1x2x2x2xf16> -> memref<1x2x2x2xf16>
    // CHECK: %0 = VPUIP.DeclareConstantTensorOp

    VPUIP.UPADMA inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>)
    // CHECK: VPUIP.UPADMA inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>)

    return
}
