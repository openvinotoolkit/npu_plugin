//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --uniquify-ops %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @UniquifyReorders(%arg0: tensor<1x16x227x227xf16>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.And(%0, %1) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[REORDER:%.*]] = IE.Reorder
    // CHECK-NOT: IE.Reorder
    // CHECK: [[RESULT:%.*]] = IE.And([[REORDER]], [[REORDER]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ReordersWithDifferentConsumerOps(%arg0: tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %cst_69 = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_90 = const.Declare tensor<1x16x1x1xf16> = dense<0.0> : tensor<1x16x1x1xf16>

    %8 = IE.HSwish(%arg0) : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16>
    %9 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %10 = IE.GroupConvolution(%9, %cst_69, %cst_90) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<16x1x3x3xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %14 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %15 = IE.Add(%10, %14) {auto_broadcast = "NUMPY"} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %15 : tensor<1x16x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST0:%.*]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST1:%.*]] = const.Declare tensor<1x16x1x1xf16>

    // CHECK: IE.HSwish
    // CHECK: [[REORDER:%.*]] = IE.Reorder
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution([[REORDER]], [[CST0]], [[CST1]])
    // CHECK-NOT: IE.Reorder
    // CHECK: IE.Add([[CONV]], [[REORDER]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TripleReordersWithDifferentConsumerOps
func @TripleReordersWithDifferentConsumerOps(%arg0: tensor<1x16x112x112xf16>) -> tensor<1x16x112x112x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>  {
    %cst_69 = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_90 = const.Declare tensor<1x16x1x1xf16> = dense<0.0> : tensor<1x16x1x1xf16>

    %8 = IE.HSwish(%arg0) : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16>
    %9 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %10 = IE.GroupConvolution(%9, %cst_69, %cst_90) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<16x1x3x3xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %14 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %15 = IE.Add(%10, %14) {auto_broadcast = "NUMPY"} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %16 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %17 = IE.And(%15, %16) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %17 : tensor<1x16x112x112x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK-DAG:   [[CST0:%.*]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST1:%.*]] = const.Declare tensor<1x16x1x1xf16>

    // CHECK: IE.HSwish
    // CHECK: [[REORDER:%.*]] = IE.Reorder
    // CHECK: [[CONV:%.*]] = IE.GroupConvolution([[REORDER]], [[CST0]], [[CST1]])
    // CHECK-NOT: IE.Reorder
    // CHECK: [[ADD:%.*]] = IE.Add([[CONV]], [[REORDER]])
    // CHECK-NOT: IE.Reorder
    // CHECK: IE.And([[ADD]], [[REORDER]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @UniquifyExpands(%arg0: tensor<1x13x227x227xf16, {order = #NHWC}>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.And(%0, %1) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[EXPAND:%.*]] = IE.Expand
    // CHECK-NOT: IE.Expand
    // CHECK: [[RESULT:%.*]] = IE.And([[EXPAND]], [[EXPAND]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @UniquifyExpandReorders(%arg0: tensor<1x13x227x227xf16>) -> tensor<2x16x227x227xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x13x227x227xf16> -> tensor<1x13x227x227xf16, {order = #NHWC}>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.And(%1, %1) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    %3 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x13x227x227xf16> -> tensor<1x13x227x227xf16, {order = #NHWC}>
    %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %5 = IE.And(%4, %4) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    %6 = IE.Concat(%2, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<2x16x227x227xf16, {order = #NHWC}>

    return %6 : tensor<2x16x227x227xf16, {order = #NHWC}>

    // CHECK:       [[REORDER:%.*]] = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x13x227x227xf16> -> tensor<1x13x227x227xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.*]] = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} :
    // CHECK-SAME:                      tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    // CHECK: [[AND0:%.*]] = IE.And([[EXPAND]], [[EXPAND]])
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[AND0]], [[AND0]])
    // CHECK: return [[CONCAT]] : tensor<2x16x227x227xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @UniquifyPermuteCast(%arg0: tensor<1x227x227x16xf16>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm=#NCHW} : tensor<1x227x227x16xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm=#NCHW} : tensor<1x227x227x16xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.And(%0, %1) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST:%.*]] = IE.PermuteCast
    // CHECK-NOT: IE.PermuteCast
    // CHECK: [[RESULT:%.*]] = IE.And([[PERMUTE_CAST]], [[PERMUTE_CAST]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @UniquifyShapeCast(%arg0: tensor<1x16x1x51529xf16, {order = #NHWC}>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 16, 227, 227]} inputs(%arg0 :tensor<1x16x1x51529xf16, {order = #NHWC}>) -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.ShapeCast {shape = [1, 16, 227, 227]} inputs(%arg0 :tensor<1x16x1x51529xf16, {order = #NHWC}>) -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.And(%0, %1) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[SHAPE_CAST:%.*]] = IE.ShapeCast
    // CHECK-NOT: IE.ShapeCast
    // CHECK: [[RESULT:%.*]] = IE.And([[SHAPE_CAST]], [[SHAPE_CAST]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutType = type tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

func @UniquifyAdd(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType
    %1 = IE.Add(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[ADD:%.*]] = IE.Add
    // CHECK-NOT: IE.Add
    // CHECK: return [[ADD]], [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutType = type tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

func @UniquifyAnd(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType
    %1 = IE.And(%arg0, %arg0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[AND:%.*]] = IE.And
    // CHECK-NOT: IE.And
    // CHECK: return [[AND]], [[AND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0082835477941176471:128>
!qElemType1 = type !quant.uniform<u8:f16, 0.017072610294117645:128>

!OutType = type tensor<1x3x512x512x!qElemType1, {order = #NHWC}>

func @UniquifyQuantizeCast(%arg0: tensor<1x3x512x512x!qElemType0, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> !OutType
    %1 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<1x3x512x512x!qElemType0, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[CAST:%.*]] = IE.QuantizeCast
    // CHECK-NOT: IE.QuantizeCast
    // CHECK: return [[CAST]], [[CAST]]
}
