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

// -----

// CHECK-LABEL: @broadcastNonSplatPreChannel
func.func @broadcastNonSplatPreChannel() -> tensor<1x9x1x1xi8> {
    %cst = const.Declare tensor<1x9x1x1xi8> = dense<[1, 2, 3]> : tensor<3xi8>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<0 : i64, 3 : i64>, #const.Reshape<[1, 9, 1, 1]>]
    return %cst : tensor<1x9x1x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x9x1x1xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1]], [[2]], [[3]], [[1]], [[2]], [[3]], [[1]], [[2]], [[3]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x9x1x1xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatPostChannel
func.func @broadcastNonSplatPostChannel() -> tensor<1x9x1x1xi8> {
    %cst = const.Declare tensor<1x9x1x1xi8> = dense<[1, 2, 3]> : tensor<3xi8>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 3 : i64>, #const.Reshape<[1, 9, 1, 1]>]
    return %cst : tensor<1x9x1x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x9x1x1xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1]], [[1]], [[1]], [[2]], [[2]], [[2]], [[3]], [[3]], [[3]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x9x1x1xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatMiddleChannel
func.func @broadcastNonSplatMiddleChannel() -> tensor<1x3x2x2xi8> {
    %cst = const.Declare tensor<1x3x2x2xi8> = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi8>, [#const.Reshape<[1, 3, 1, 2]>, #const.Broadcast<2 : i64, 2 : i64>]
    return %cst : tensor<1x3x2x2xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x3x2x2xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1, 2], [1, 2]],
    // CHECK-SAME{LITERAL}:       [[3, 4], [3, 4]],
    // CHECK-SAME{LITERAL}:       [[5, 6], [5, 6]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x3x2x2xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatMiddleChannelNone1
func.func @broadcastNonSplatMiddleChannelNone1() -> tensor<1x3x4x2xi8> {
    %cst = const.Declare tensor<1x3x4x2xi8> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]> : tensor<12xi8>, [#const.Reshape<[1, 3, 2, 2]>, #const.Broadcast<2 : i64, 4 : i64>]
    return %cst : tensor<1x3x4x2xi8>
    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x3x4x2xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1, 2], [3, 4], [1, 2], [3, 4]],
    // CHECK-SAME{LITERAL}:       [[5, 6], [7, 8], [5, 6], [7, 8]],
    // CHECK-SAME{LITERAL}:       [[9, 10], [11, 12], [9, 10], [11, 12]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x3x4x2xi8>
    // CHECK:               return [[CST]]
}
