//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: memref<1x3x16x16xf32, #NHWC>) -> memref<1x3x16x16xf32, #NHWC> {
    %0 = const.Declare memref<1x3x16x16xf32, #NHWC> =
        dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x3x16x16xf32, #NHWC>) -> memref<1x3x16x16xf32, #NHWC>

    %2 = VPUIP.Copy
        inputs(%1 : memref<1x3x16x16xf32, #NHWC>)
        outputs(%arg0 : memref<1x3x16x16xf32, #NHWC>)
        -> memref<1x3x16x16xf32, #NHWC>

    return %2 : memref<1x3x16x16xf32, #NHWC>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x16x16xf32, #NHWC>
    // CHECK-SAME:       dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x16x16xf32, #NHWC>) outputs(%arg0 : memref<1x3x16x16xf32, #NHWC>)
    // CHECK:       return [[VAR0]] : memref<1x3x16x16xf32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FuseGenericReshapes
func.func @FuseGenericReshapes(%arg0: memref<1x3x16x2xf32>) -> memref<1x3x16x2xf32> {
    %0 = const.Declare memref<1x3x2x16xf32> =
        dense<1.000000e+00> : tensor<1x3x2x16xf32>

    %1 = memref.alloc() : memref<1x3x2x16xf32>
    %2 = VPUIP.Copy
        inputs(%0 : memref<1x3x2x16xf32>)
        outputs(%1 : memref<1x3x2x16xf32>)
        -> memref<1x3x2x16xf32>

    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x3x2x16xf32>) -> memref<1x3x4x8xf32>
    %4 = VPUIP.GenericReshape inputs(%3 : memref<1x3x4x8xf32>) -> memref<1x3x16x2xf32>

    %5 = VPUIP.Copy
        inputs(%4 : memref<1x3x16x2xf32>)
        outputs(%arg0 : memref<1x3x16x2xf32>)
        -> memref<1x3x16x2xf32>

    return %5 : memref<1x3x16x2xf32>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x2x16xf32> = dense<1.000000e+00> : tensor<1x3x2x16xf32>
    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x3x2x16xf32>
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x2x16xf32>) outputs([[VAR0]] : memref<1x3x2x16xf32>)
    // CHECK:       [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x3x2x16xf32>) -> memref<1x3x16x2xf32>
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x3x16x2xf32>) outputs(%arg0 : memref<1x3x16x2xf32>)
    // CHECK:       return [[VAR3]] : memref<1x3x16x2xf32>
}
