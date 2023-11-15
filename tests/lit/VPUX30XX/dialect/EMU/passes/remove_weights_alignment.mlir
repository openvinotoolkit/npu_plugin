//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --remove-weights-alignment --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveWeightsAlignmentCMCONV
func.func @RemoveWeightsAlignmentCMCONV(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<32x1x1x32xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<32x1x1x32xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = EMU.NCEClusterTask {
            kernel_padding = [1, 0, 1, 0],
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            rawFilterShape = [32, 3, 3, 3],
            task_type = #VPUIP.nce_task_type<CMCONV>
        }
        input(%arg0 : tensor<1x3x224x224xf16>)
        weights(%cst0 : tensor<32x1x1x32xf16, {order = #NHWC}>)
        weight_table(%cst1 : tensor<32x1x1x4xsi32>)
        -> tensor<1x32x112x112xf16, {order = #NHWC}>
        PPE : { PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64} }

    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<32x3x3x3xf16>
    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    
    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [1, 0, 1, 0], kernel_size = [3, 3], kernel_strides = [2, 2], rawFilterShape = [32, 3, 3, 3], task_type = #VPUIP.nce_task_type<CMCONV>
    // CHECK-SAME:      input(%arg0 : tensor<1x3x224x224xf16>)
    // CHECK-SAME:      weights([[CST0]] : tensor<32x3x3x3xf16>)
    // CHECK-SAME:      weight_table([[CST1]] : tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x32x112x112xf16, {order = #NHWC}>
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}


    // CHECK:       return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveWeightsAlignmentDWCONVConst
func.func @RemoveWeightsAlignmentDWCONVConst(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = EMU.NCEClusterTask {
            kernel_padding = [1, 1, 1, 1],
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            rawFilterShape = [32, 1, 3, 3],
            task_type = #VPUIP.nce_task_type<DWCONV>
        }
        input(%arg0 : tensor<1x32x112x112xf16, {order = #NHWC}>)
        weights(%cst0 : tensor<32x16x1x1xf16, {order = #NHWC}>)
        weight_table(%cst1 : tensor<32x1x1x4xsi32>)
        -> tensor<1x32x112x112xf16, {order = #NHWC}>
        PPE : { PPETask <LRELU> {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0} }

    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<32x3x3xf16>
    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    
    // CHECK:       [[VAL0:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [1, 1, 1, 1], kernel_size = [3, 3], kernel_strides = [1, 1], rawFilterShape = [32, 1, 3, 3], task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:      input(%arg0 : tensor<1x32x112x112xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights([[CST0]] : tensor<32x3x3xf16>)
    // CHECK-SAME:      weight_table([[CST1]] : tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x32x112x112xf16, {order = #NHWC}>
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}


    // CHECK:       return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @RemoveWeightsAlignmentDWCONVNonConst
func.func @RemoveWeightsAlignmentDWCONVNonConst(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<32x1x3x3xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst1 = const.Declare tensor<32x7x1x1xf16> = dense<0.0e+00> : tensor<32x7x1x1xf16>

    %0 = IE.Reshape(%arg1) { shape_value = [32, 9, 1, 1] } :
        tensor<32x1x3x3xf16, {order = #NHWC}> -> tensor<32x9x1x1xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<32x9x1x1xf16> -> tensor<32x9x1x1xf16>
    %2 = IE.Concat(%1, %cst1) {per_axis = #IE.Concat<axis = 1>} :
        tensor<32x9x1x1xf16>, tensor<32x7x1x1xf16> -> tensor<32x16x1x1xf16>
    %3 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} :
        tensor<32x16x1x1xf16> -> tensor<32x16x1x1xf16, {order = #NHWC}>

    %4 = EMU.NCEClusterTask {
            kernel_padding = [1, 1, 1, 1],
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            rawFilterShape = [32, 1, 3, 3],
            task_type = #VPUIP.nce_task_type<DWCONV>
        }
        input(%arg0 : tensor<1x32x112x112xf16, {order = #NHWC}>)
        weights(%3 : tensor<32x16x1x1xf16, {order = #NHWC}>)
        weight_table(%cst0 : tensor<32x1x1x4xsi32>)
        -> tensor<1x32x112x112xf16, {order = #NHWC}>
        PPE : { PPETask <LRELU> {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0} }

    return %4 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<32x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<32x1x3x3xf16, {order = #NHWC}> -> tensor<32x1x3x3xf16>

    // CHECK:       [[VAL1:%.+]] = IE.AffineReshape([[VAL0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2]], shape_value = [32, 3, 3]} : tensor<32x1x3x3xf16> -> tensor<32x3x3xf16>
    
    // CHECK:       [[VAL2:%.+]] = EMU.NCEClusterTask
    // CHECK-SAME:      kernel_padding = [1, 1, 1, 1], kernel_size = [3, 3], kernel_strides = [1, 1], rawFilterShape = [32, 1, 3, 3], task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:      input(%arg0 : tensor<1x32x112x112xf16, {order = #NHWC}>)
    // CHECK-SAME:      weights([[VAL1]] : tensor<32x3x3xf16>)
    // CHECK-SAME:      weight_table([[CST0]] : tensor<32x1x1x4xsi32>)
    // CHECK-SAME:      -> tensor<1x32x112x112xf16, {order = #NHWC}>
    // CHECK:           PPETask <LRELU> {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}


    // CHECK:       return [[VAL2]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}
