//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --constant-folding %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType = type !quant.uniform<u8:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func @ConstFold() -> memref<16x3x1x1xf16, #YXOI> {
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

!qElemType = type !quant.uniform<u8:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func @QuantConstFold() -> memref<16x3x1x1x!qElemType, #YXOI> {
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
