//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-permute-post-eltwise --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8:f16, 0.001>
!qElemType1 = type !quant.uniform<u8:f16, 0.002>
!qElemType2 = type !quant.uniform<u8:f16, 0.003>
!qElemType3 = type !quant.uniform<u8:f16, 0.004>
!qElemType4 = type !quant.uniform<u8:f16, 0.005>


// CHECK-LABEL: @MovePermutePostAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256xf16>
func @MovePermutePostAdd(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = "NUMPY"} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERMUTE_CAST_1]] : tensor<1x256x3x256xf16, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERMUTE_CAST_2]] : tensor<1x256x3x256xf16, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST_1]], [[SHAPE_CAST_2]]) {auto_broadcast = "NUMPY"} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 3, 256]} inputs([[ADD]] : tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddWithOneInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x256x256xf16>
func @MovePermutePostAddWithOneInput(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %0) {auto_broadcast = "NUMPY"} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERMUTE_CAST]] : tensor<1x256x3x256xf16, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[SHAPE_CAST]]) {auto_broadcast = "NUMPY"} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 3, 256]} inputs([[ADD]] : tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddWithQuantizeCast
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256x!qElemType0>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256x!qElemType1>
func @MovePermutePostAddWithQuantizeCast(%arg0: tensor<1x3x256x256x!qElemType0>, %arg1: tensor<1x3x256x256x!qElemType1>)
        -> tensor<1x3x256x256x!qElemType0, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %permute1 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256x!qElemType0> -> tensor<1x3x256x256x!qElemType0, {order = #NHWC}>
    %qc1 = IE.QuantizeCast(%permute1) {dstElemType = !qElemType2} : tensor<1x3x256x256x!qElemType0, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType2, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %permute2 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256x!qElemType1> -> tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
    %qc2 = IE.QuantizeCast(%permute2) {dstElemType = !qElemType3} : tensor<1x3x256x256x!qElemType1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType3, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %add = IE.Add(%qc1, %qc2) {auto_broadcast = "NUMPY"} : tensor<1x3x256x256x!qElemType2, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x256x256x!qElemType3, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType4, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %qc3 = IE.QuantizeCast(%add) {dstElemType = !qElemType0} : tensor<1x3x256x256x!qElemType4, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType0, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    return %qc3 : tensor<1x3x256x256x!qElemType0, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK:       [[PERM_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256x!qElemType0>
    // CHECK-SAME:          -> tensor<1x256x3x256x!qElemType0, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERM_CAST_1]] : tensor<1x256x3x256x!qElemType0, {order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType0, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_1:%.+]] = IE.QuantizeCast([[SHAPE_CAST_1]]) {dstElemType = !qElemType2} : tensor<1x3x256x256x!qElemType0, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType2, {order = #NHWC}>
    // CHECK:       [[PERM_CAST_2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256x!qElemType1>
    // CHECK-SAME:          -> tensor<1x256x3x256x!qElemType1, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERM_CAST_2]] : tensor<1x256x3x256x!qElemType1, {order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_2:%.+]] = IE.QuantizeCast([[SHAPE_CAST_2]]) {dstElemType = !qElemType3} : tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType3, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[QUANT_CAST_1]], [[QUANT_CAST_2]]) {auto_broadcast = "NUMPY"} : tensor<1x3x256x256x!qElemType2, {order = #NHWC}>, tensor<1x3x256x256x!qElemType3, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType4, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_OUT:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType0} : tensor<1x3x256x256x!qElemType4, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType0, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 3, 256]} inputs([[QUANT_CAST_OUT]] : tensor<1x3x256x256x!qElemType0, {order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x256x3x256x!qElemType0, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256x!qElemType0, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType0, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256x!qElemType0, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddWithSameLayout
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256xf16, {order = #NHWC}>
func @MovePermutePostAddWithSameLayout(%arg0: tensor<1x3x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x3xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = "NUMPY"} : tensor<1x256x256x3xf16, {order = #NHWC}>, tensor<1x256x256x3xf16, {order = #NHWC}>
            -> tensor<1x256x256x3xf16, {order = #NHWC}>
    return %2 : tensor<1x256x256x3xf16, {order = #NHWC}>


    // CHECK:       [[SHAPE_CAST_1:%.+]]  = IE.ShapeCast {shape = [1, 256, 256, 3]} inputs([[INPUT1]] : tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]]  = IE.ShapeCast {shape = [1, 256, 256, 3]} inputs([[INPUT2]] : tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]]  = IE.Add([[SHAPE_CAST_1]], [[SHAPE_CAST_2]]) {auto_broadcast = "NUMPY"} : tensor<1x256x256x3xf16, {order = #NHWC}>, tensor<1x256x256x3xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]]  = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[ADD]] : tensor<1x256x256x3xf16, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]]  = IE.MemPermute([[SHAPE_CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x256x256x3xf16, {order = #NHWC}>
}
