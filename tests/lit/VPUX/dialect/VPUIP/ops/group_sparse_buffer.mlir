//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  func.func @Fold([[ARG0:%.+]]: memref<1x128x16x16xf16>, [[ARG1:%.+]]: memref<1x128x16x16xi1>
func.func @Fold(%arg0: memref<1x128x16x16xf16>, %arg1: memref<1x128x16x16xi1>) -> (memref<1x128x16x16xf16>, memref<1x128x16x16xi1>) {
    %0 = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !VPUIP.SparseBuffer<data=memref<1x128x16x16xf16>, sparsity_map=memref<1x128x16x16xi1>>
    %1, %2 = VPUIP.UngroupSparseBuffer(%0) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> memref<1x128x16x16xf16>, memref<1x128x16x16xi1>
    return %1, %2 : memref<1x128x16x16xf16>, memref<1x128x16x16xi1>

    // CHECK-NOT:  VPUIP.GroupSparseBuffer
    // CHECK-NOT:  VPUIP.UngroupSparseBuffer
    // CHECK:      return [[ARG0]], [[ARG1]]
}
