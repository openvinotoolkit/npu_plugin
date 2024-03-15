//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-mat-mul %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @UnrollMatMul
// CHECK-SAME:   [[LHS_1:%arg0]]: tensor<16x96xf32>,
// CHECK-SAME:   [[RHS_1:%arg1]]: tensor<1x32x64xf32>,
// CHECK-SAME:   [[RHS_2:%arg2]]: tensor<1x32x64xf32>,
// CHECK-SAME:   [[RHS_3:%arg3]]: tensor<1x32x64xf32>
func.func @UnrollMatMul(%LHS_1: tensor<16x96xf32>,
                        %RHS_1: tensor<1x32x64xf32>,
                        %RHS_2: tensor<1x32x64xf32>,
                        %RHS_3: tensor<1x32x64xf32>) -> tensor<16x64xf32> {
    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x32x64xf32>, tensor<1x32x64xf32>, tensor<1x32x64xf32> -> tensor<3x32x64xf32>
    // CHECK-NOT:   IE.Concat

    %DST_SHAPE = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>
    %RESHAPE_RHS = IE.Reshape(%CONCAT_RHS, %DST_SHAPE) : tensor<3x32x64xf32>, tensor<2xsi64> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS_1:%.*]] = IE.Reshape([[RHS_1]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>
    // CHECK:   [[RESHAPE_RHS_2:%.*]] = IE.Reshape([[RHS_2]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>
    // CHECK:   [[RESHAPE_RHS_3:%.*]] = IE.Reshape([[RHS_3]]) {
    // CHECK-SAME:      shape_value = [32, 64]
    // CHECK-SAME:  } : tensor<1x32x64xf32> -> tensor<32x64xf32>

    %GEMM = IE.MatMul(%LHS_1, %RESHAPE_RHS) : tensor<16x96xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
    // CHECK:   [[LHS_SLICE_1:%.*]] = IE.Slice [[LHS_1]] [0, 0] [16, 32]
    // CHECK:   [[LHS_SLICE_2:%.*]] = IE.Slice [[LHS_1]] [0, 32] [16, 32]
    // CHECK:   [[LHS_SLICE_3:%.*]] = IE.Slice [[LHS_1]] [0, 64] [16, 32]

    // CHECK:   [[GEMM_1:%.*]] = IE.MatMul([[LHS_SLICE_1]], [[RESHAPE_RHS_1]])
    // CHECK:   [[GEMM_2:%.*]] = IE.MatMul([[LHS_SLICE_2]], [[RESHAPE_RHS_2]])
    // CHECK:   [[GEMM_3:%.*]] = IE.MatMul([[LHS_SLICE_3]], [[RESHAPE_RHS_3]])

    // CHECK:   [[ADD_1:%.*]] = IE.Add([[GEMM_1]], [[GEMM_2]])
    // CHECK:   [[ADD_2:%.*]] = IE.Add([[ADD_1]], [[GEMM_3]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[ADD_2]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithTransposeA
// CHECK-SAME:   [[LHS:%.*]]: tensor<96x16xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<96x64xf32>
func.func @SkipMatMulWithTransposeA(%LHS: tensor<96x16xf32>, %RHS: tensor<96x64xf32>) -> tensor<16x64xf32> {
    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_a} : tensor<96x16xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithTransposeB
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<64x96xf32>
func.func @SkipMatMulWithTransposeB(%LHS: tensor<16x96xf32>, %RHS: tensor<64x96xf32>) -> tensor<16x64xf32> {
    %GEMM = IE.MatMul(%LHS, %RHS) {transpose_b} : tensor<16x96xf32>, tensor<64x96xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWith3dLHS
// CHECK-SAME:   [[LHS:%.*]]: tensor<1x16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<96x64xf32>
func.func @SkipMatMulWith3dLHS(%LHS: tensor<1x16x96xf32>, %RHS: tensor<96x64xf32>) -> tensor<1x16x64xf32> {
    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<1x16x96xf32>, tensor<96x64xf32> -> tensor<1x16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RHS]])

    return %GEMM : tensor<1x16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<1x16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWith3dRHS
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<1x96x64xf32>
func.func @SkipMatMulWith3dRHS(%LHS: tensor<16x96xf32>, %RHS: tensor<1x96x64xf32>) -> tensor<1x16x64xf32> {
    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<16x96xf32>, tensor<1x96x64xf32> -> tensor<1x16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RHS]])

    return %GEMM : tensor<1x16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<1x16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithoutReshape
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<96x64xf32>
func.func @SkipMatMulWithoutReshape(%LHS: tensor<16x96xf32>, %RHS: tensor<96x64xf32>) -> tensor<16x64xf32> {
    %GEMM = IE.MatMul(%LHS, %RHS) : tensor<16x96xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithUnsupportedReshape
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<2x96x32xf32>
func.func @SkipMatMulWithUnsupportedReshape(%LHS: tensor<16x96xf32>, %RHS: tensor<2x96x32xf32>) -> tensor<16x64xf32> {
    %DST_SHAPE = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>
    // CHECK:   [[DST_SHAPE:%.*]] = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>

    %RESHAPE_RHS = IE.Reshape(%RHS, %DST_SHAPE) : tensor<2x96x32xf32>, tensor<2xsi64> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.*]] = IE.Reshape([[RHS]], [[DST_SHAPE]]) : tensor<2x96x32xf32>, tensor<2xsi64> -> tensor<96x64xf32>

    %GEMM = IE.MatMul(%LHS, %RESHAPE_RHS) : tensor<16x96xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RESHAPE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithoutConcat
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS:%.*]]: tensor<3x32x64xf32>
func.func @SkipMatMulWithoutConcat(%LHS: tensor<16x96xf32>, %RHS: tensor<3x32x64xf32>) -> tensor<16x64xf32> {
    %DST_SHAPE = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>
    // CHECK:   [[DST_SHAPE:%.*]] = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>

    %RESHAPE_RHS = IE.Reshape(%RHS, %DST_SHAPE) : tensor<3x32x64xf32>, tensor<2xsi64> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.*]] = IE.Reshape([[RHS]], [[DST_SHAPE]]) : tensor<3x32x64xf32>, tensor<2xsi64> -> tensor<96x64xf32>

    %GEMM = IE.MatMul(%LHS, %RESHAPE_RHS) : tensor<16x96xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RESHAPE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: @SkipMatMulWithUnsupportedConcat
// CHECK-SAME:   [[LHS:%.*]]: tensor<16x96xf32>
// CHECK-SAME:   [[RHS_0:%.*]]: tensor<3x16x64xf32>,
// CHECK-SAME:   [[RHS_1:%.*]]: tensor<3x16x64xf32>
func.func @SkipMatMulWithUnsupportedConcat(%LHS: tensor<16x96xf32>,
                                           %RHS_0: tensor<3x16x64xf32>,
                                           %RHS_1: tensor<3x16x64xf32>) -> tensor<16x64xf32> {
    %DST_SHAPE = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>
    // CHECK-DAG:   [[DST_SHAPE:%.*]] = const.Declare tensor<2xsi64> = dense<[96, 64]> : tensor<2xsi64>

    %CONCAT_RHS = IE.Concat(%RHS_0, %RHS_1) {
        per_axis = #IE.Concat<axis = 1 : i64>
    } : tensor<3x16x64xf32>, tensor<3x16x64xf32> -> tensor<3x32x64xf32>
    // CHECK:   [[CONCAT_RHS:%.*]] = IE.Concat([[RHS_0]], [[RHS_1]])

    %RESHAPE_RHS = IE.Reshape(%CONCAT_RHS, %DST_SHAPE) : tensor<3x32x64xf32>, tensor<2xsi64> -> tensor<96x64xf32>
    // CHECK:   [[RESHAPE_RHS:%.*]] = IE.Reshape([[CONCAT_RHS]], [[DST_SHAPE]]) : tensor<3x32x64xf32>, tensor<2xsi64> -> tensor<96x64xf32>

    %GEMM = IE.MatMul(%LHS, %RESHAPE_RHS) : tensor<16x96xf32>, tensor<96x64xf32> -> tensor<16x64xf32>
    // CHECK:   [[GEMM:%.*]] = IE.MatMul([[LHS]], [[RESHAPE_RHS]])

    return %GEMM : tensor<16x64xf32>
    // CHECK:   return [[GEMM]] : tensor<16x64xf32>
}
