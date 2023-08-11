//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --constant-folding %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func.func @ConstFold() -> memref<16x3x1x1xf16, #YXOI> {
    %0 = const.Declare memref<16x3x1x1xf16, #YXOI> =
        dense<-1.0> : tensor<16x3x1x1xf32>,
        [
            #const.ConvertElemType<f16>,
            #const.ConvertElemType<ui8>,
            #const.QuantCast<!qElemType>,
            #const.Dequantize,
            #const.Reorder<#YXOI>
        ]

    return %0 : memref<16x3x1x1xf16, #YXOI>

    // CHECK:       [[CST:%.*]] = const.Declare memref<16x3x1x1xf16, #YXOI>
    // CHECK-SAME:       dense<
    // CHECK-SAME:       tensor<16x3x1x1xf16
    // CHECK-SAME:       {order = #YXOI}>
    // CHECK:       return [[CST]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func.func @QuantConstFold() -> memref<16x3x1x1x!qElemType, #YXOI> {
    %0 = const.Declare memref<16x3x1x1x!qElemType, #YXOI> =
        dense<129> : tensor<16x3x1x1xui8>,
        [
            #const.QuantCast<!qElemType>,
            #const.Reorder<#YXOI>
        ]

    return %0 : memref<16x3x1x1x!qElemType, #YXOI>

    // CHECK:       [[CST:%.*]] = const.Declare memref<16x3x1x1x!qElemType, #YXOI>
    // CHECK-SAME:       dense<
    // CHECK-SAME:       tensor<16x3x1x1xui8
    // CHECK-SAME:       {order = #YXOI}>
    // CHECK:       return [[CST]]
}

// -----

func.func @I1SubviewConstFoldSplat() -> memref<1x16x3x3xi1> {
    %cst = const.Declare memref<1x16x3x3xi1> =
        dense<true> : tensor<1x32x3x3xi1>,
        [
            #const.SubView<[0, 16, 0, 0], [1, 16, 3, 3]>
        ]

    return %cst : memref<1x16x3x3xi1>

    // CHECK:   [[CST:%.*]] = const.Declare memref<1x16x3x3xi1> = dense<true> : tensor<1x16x3x3xi1>
    // CHECK:   return [[CST]]
}

// -----

func.func @I1SubviewConstFoldNonSplat() -> memref<1x16x1x1xi1> {
    %cst = const.Declare memref<1x16x1x1xi1> =
        dense<[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],
                [[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],
                [[0]],[[1]],[[0]],[[1]],[[0]],[[1]],[[0]],[[1]],
                [[1]],[[0]],[[1]],[[0]],[[1]],[[0]],[[1]],[[0]]]]> : tensor<1x32x1x1xi1>,
        [
            #const.SubView<[0, 16, 0, 0], [1, 16, 1, 1]>
        ]

    return %cst : memref<1x16x1x1xi1>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x16x1x1xi1> = dense<
    // CHECK-SAME{LITERAL}:     [[[[false]], [[true]], [[false]], [[true]], [[false]], [[true]], [[false]], [[true]],
    // CHECK-SAME{LITERAL}:       [[true]], [[false]], [[true]], [[false]], [[true]], [[false]], [[true]], [[false]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x16x1x1xi1>
    // CHECK:               return [[CST]]
}
