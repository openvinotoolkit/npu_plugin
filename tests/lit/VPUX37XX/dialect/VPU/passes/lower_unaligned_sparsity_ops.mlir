//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file  --lower-sparsity-ops="fake-sparsify=false" %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputType = type tensor<1x4x120x110x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = type tensor<1x4x120x110xi1, {order = #NHWC}>
!sparseType = type !VPU.SparseTensor<data=!inputType, sparsity_map=!smType>
!weightsType = type tensor<64x1x1x16xf16, {order = #NHWC}>
!outputType = type tensor<1x64x120x110x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>

module attributes {VPU.arch = "VPUX37XX"} {

func @LowerSparsifyOpUniformQuantUnalignedShape(%arg0: !inputType, %wt: tensor<64x1x1x4xsi32>, %weights: !weightsType) -> !outputType {
    %0 = VPU.Sparsify(%arg0) : !inputType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
            rawFilterShape = [64, 4, 1, 1], 
            strides = [1, 1]
        } -> !outputType 

    return %1 : !outputType

    // CHECK:       [[VAL0:%.+]] = VPU.AffineReshape(%arg0) 
    // CHECK-SAME{LITERAL}:              dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 48, 25, 44]}
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:                   op_type = "ADD"
    // CHECK:                        quant_mult = [16384]
    // CHECK-SAME:                   quant_post_shift = 0
    // CHECK-SAME:                   quant_shift = [29]
    // CHECK-SAME:                   !VPU.SparseTensor
    // CHECK:       [[VAL2:%.+]] = VPU.AffineReshape([[VAL1]]) 
    // CHECK-SAME{LITERAL}:              dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 4, 120, 110]}
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], %arg2, %arg1)
    // CHECK:       return [[VAL3]]
}
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputType = type tensor<1x4x31x31xf16, {order = #NHWC}>
!smType = type tensor<1x4x31x31xi1, {order = #NHWC}>
!sparseType = type !VPU.SparseTensor<data=!inputType, sparsity_map=!smType>
!weightsType = type tensor<64x1x1x16xf16, {order = #NHWC}>
!outputType = type tensor<1x64x31x31xf16, {order = #NHWC}>

module attributes {VPU.arch = "VPUX37XX"} {

func @LowerSparsifyOpUniformQuantUnalignedUnaliqoutShape(%arg0: !inputType, %wt: tensor<64x1x1x4xsi32>, %weights: !weightsType) -> !outputType {
    %0 = VPU.Sparsify(%arg0) : !inputType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
            rawFilterShape = [64, 4, 1, 1], 
            strides = [1, 1]
        } -> !outputType 

    return %1 : !outputType

    // CHECK:       [[VAL0:%.+]] = VPU.Expand(%arg0) 
    // CHECK-SAME:                   pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:                   op_type = "ADD"
    // CHECK-SAME:                   quant_scale = [5.000000e-01]
    // CHECK-SAME:                   !VPU.SparseTensor
    // CHECK:       [[VAL2:%.+]] = VPU.Slice [[VAL1]] [0, 0, 0, 0] [1, 4, 31, 31]
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]], %arg2, %arg1)
    // CHECK:       return [[VAL3]]
}
}
