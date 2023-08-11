//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --upstream-slice %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @UpstreamSlice
!qElemType0 = !quant.uniform<u8:f16, 0.0078431377223893706:128>
!qElemType1 = !quant.uniform<i8:f16, 0.0078431377223893706>
!qElemType2 = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType3 = !quant.uniform<u8:f16, 0.015686274509803921>
!qElemType4 = !quant.uniform<u8:f16, 0.031372549019607843>
func.func @UpstreamSlice(%arg0: tensor<1x16x64x4xf16>, %arg1: tensor<1x16x17x2xf16>) -> tensor<1x32x16x4xf16> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_0 = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_1 = const.Declare tensor<32x16x3x1x!qElemType0> = dense<1.0> : tensor<32x16x3x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType1>, #const.QuantCast<>, #const.ConvertElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]
    %cst_2 = const.Declare tensor<64x16x3x1x!qElemType0> = dense<1.0> : tensor<64x16x3x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType1>, #const.QuantCast<>, #const.ConvertElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x64x4xf16> -> tensor<1x16x64x4x!qElemType2>
    %1 = IE.Convolution(%0, %cst_1, %cst) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x64x4x!qElemType2>, tensor<32x16x3x1x!qElemType0>, tensor<1x32x1x1xf16> -> tensor<1x32x64x4x!qElemType3>
    %2 = IE.Slice %1 [0, 0, 47, 0] [1, 32, 17, 4] : tensor<1x32x64x4x!qElemType3> to tensor<1x32x17x4x!qElemType3>
    %3 = IE.Quantize(%arg1) {dstElemType = !qElemType2} : tensor<1x16x17x2xf16> -> tensor<1x16x17x2x!qElemType2>
    %4 = IE.Convolution(%3, %cst_2, %cst_0) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x17x2x!qElemType2>, tensor<64x16x3x1x!qElemType0>, tensor<1x64x1x1xf16> -> tensor<1x64x17x2x!qElemType3>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x64x17x2x!qElemType3> -> tensor<1x64x17x2xf16>
    %6 = IE.Reshape(%5) {shape_value = [1, 32, 17, 4]} : tensor<1x64x17x2xf16> -> tensor<1x32x17x4xf16>
    %7 = IE.Quantize(%6) {dstElemType = !qElemType3} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType3>
    %8 = IE.Add(%2, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x17x4x!qElemType3>, tensor<1x32x17x4x!qElemType3> -> tensor<1x32x17x4x!qElemType4>
    %9 = IE.Dequantize(%8) {dstElemType = f16} : tensor<1x32x17x4x!qElemType4> -> tensor<1x32x17x4xf16>
    %10 = IE.Slice %9 [0, 0, 1, 0] [1, 32, 16, 4] : tensor<1x32x17x4xf16> to tensor<1x32x16x4xf16>
    return %10 : tensor<1x32x16x4xf16>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1x32x1x1xf16>
    // CHECK-DAG:   %[[CST0:.*]] = const.Declare tensor<1x64x1x1xf16>
    // CHECK-DAG:   %[[CST1:.*]] = const.Declare tensor<32x16x3x1x!qElemType0>
    // CHECK-DAG:   %[[CST2:.*]] = const.Declare tensor<64x16x3x1x!qElemType0>

    // CHECK:       %[[QUANT0:.*]] = IE.Quantize(%arg0)
    // CHECK:       %[[CONV0:.*]] = IE.Convolution(%[[QUANT0]], %[[CST1]], %[[CST]])
    // CHECK:       %[[QUANT1:.*]] = IE.Quantize(%arg1)
    // CHECK:       %[[CONV1:.*]] = IE.Convolution(%[[QUANT1]], %[[CST2]], %[[CST0]])
    // CHECK:       %[[DEQUANT:.*]] = IE.Dequantize(%[[CONV1]])
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%[[DEQUANT]])
    // CHECK:       %[[SLICE1:.*]] = IE.Slice %[[RESHAPE]]
    // CHECK-SAME:    [0, 0, 1, 0] [1, 32, 16, 4] : tensor<1x32x17x4xf16> to tensor<1x32x16x4xf16>
    // CHECK:       %[[QUANT:.*]] = IE.Quantize(%[[SLICE1]])
    // CHECK:       %[[SLICE0:.*]] = IE.Slice %[[CONV0]]
    // CHECK-SAME:    [0, 0, 48, 0] [1, 32, 16, 4] : tensor<1x32x64x4x!qElemType3> to tensor<1x32x16x4x!qElemType3>
    // CHECK:       %[[ADD:.*]] = IE.Add(%[[SLICE0]], %[[QUANT]])
    // CHECK-SAME:    {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x4x!qElemType3>, tensor<1x32x16x4x!qElemType3> -> tensor<1x32x16x4x!qElemType4>
    // CHECK:       %[[DEQUANT1:.*]] = IE.Dequantize(%[[ADD]])
    // CHECK-SAME:   {dstElemType = f16} : tensor<1x32x16x4x!qElemType4> -> tensor<1x32x16x4xf16>
    // CHECK:       return  %[[DEQUANT1]]
}

// -----

// CHECK-LABEL: @UpstreamStridedSlice
!qElemType0 = !quant.uniform<u8:f16, 0.0078431377223893706:128>
!qElemType1 = !quant.uniform<i8:f16, 0.0078431377223893706>
!qElemType2 = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType3 = !quant.uniform<u8:f16, 0.015686274509803921>
!qElemType4 = !quant.uniform<u8:f16, 0.031372549019607843>
func.func @UpstreamStridedSlice(%arg0: tensor<1x16x64x4xf16>, %arg1: tensor<1x16x17x2xf16>) -> tensor<1x32x16x4xf16> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_0 = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_1 = const.Declare tensor<32x16x3x1x!qElemType0> = dense<1.0> : tensor<32x16x3x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType1>, #const.QuantCast<>, #const.ConvertElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]
    %cst_2 = const.Declare tensor<64x16x3x1x!qElemType0> = dense<1.0> : tensor<64x16x3x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<si8>, #const.QuantCast<!qElemType1>, #const.QuantCast<>, #const.ConvertElemType<i32>, #const.Add<1.280000e+02 : f64>, #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x16x64x4xf16> -> tensor<1x16x64x4x!qElemType2>
    %1 = IE.Convolution(%0, %cst_1, %cst) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x64x4x!qElemType2>, tensor<32x16x3x1x!qElemType0>, tensor<1x32x1x1xf16> -> tensor<1x32x64x4x!qElemType3>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x32x64x4x!qElemType3> -> tensor<1x32x64x4xf16>
    %3 = IE.StridedSlice(%2) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 64, 4], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x32x64x4xf16> -> tensor<1x32x17x4xf16>
    %4 = IE.Quantize(%3) {dstElemType = !qElemType3} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType3>
    %5 = IE.Quantize(%arg1) {dstElemType = !qElemType2} : tensor<1x16x17x2xf16> -> tensor<1x16x17x2x!qElemType2>
    %6 = IE.Convolution(%5, %cst_2, %cst_0) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]} : tensor<1x16x17x2x!qElemType2>, tensor<64x16x3x1x!qElemType0>, tensor<1x64x1x1xf16> -> tensor<1x64x17x2x!qElemType3>
    %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x64x17x2x!qElemType3> -> tensor<1x64x17x2xf16>
    %8 = IE.Reshape(%7) {shape_value = [1, 32, 17, 4]} : tensor<1x64x17x2xf16> -> tensor<1x32x17x4xf16>
    %9 = IE.Quantize(%8) {dstElemType = !qElemType3} : tensor<1x32x17x4xf16> -> tensor<1x32x17x4x!qElemType3>
    %10 = IE.Add(%4, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x17x4x!qElemType3>, tensor<1x32x17x4x!qElemType3> -> tensor<1x32x17x4x!qElemType4>
    %11 = IE.Dequantize(%10) {dstElemType = f16} : tensor<1x32x17x4x!qElemType4> -> tensor<1x32x17x4xf16>
    %12 = IE.StridedSlice(%11) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 17, 4], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]} : tensor<1x32x17x4xf16> -> tensor<1x32x16x4xf16>
    return %12 : tensor<1x32x16x4xf16>

    // CHECK-DAG:   %[[CST:.*]] = const.Declare tensor<1x32x1x1xf16>
    // CHECK-DAG:   %[[CST0:.*]] = const.Declare tensor<1x64x1x1xf16>
    // CHECK-DAG:   %[[CST1:.*]] = const.Declare tensor<32x16x3x1x!qElemType0>
    // CHECK-DAG:   %[[CST2:.*]] = const.Declare tensor<64x16x3x1x!qElemType0>

    // CHECK:       %[[QUANT0:.*]] = IE.Quantize(%arg0)
    // CHECK:       %[[CONV0:.*]] = IE.Convolution(%[[QUANT0]], %[[CST1]], %[[CST]])
    // CHECK:       %[[DEQUANT0:.*]] = IE.Dequantize(%[[CONV0]])
    // CHECK:       %[[SS1:.*]] = IE.StridedSlice(%[[DEQUANT0]])
    // CHECK-SAME:   {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 47, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 64, 4], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]}
    // CHECK:       %[[QUANT0:.*]] = IE.Quantize(%[[SS1]])
    // CHECK:       %[[QUANT1:.*]] = IE.Quantize(%arg1)
    // CHECK:       %[[CONV1:.*]] = IE.Convolution(%[[QUANT1]], %[[CST2]], %[[CST0]])
    // CHECK:       %[[DEQUANT:.*]] = IE.Dequantize(%[[CONV1]])
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%[[DEQUANT]])
    // CHECK:       %[[QUANT2:.*]] = IE.Quantize(%[[RESHAPE]])
    // CHECK:       %[[ADD:.*]] = IE.Add(%[[QUANT0]], %[[QUANT2]])
    // CHECK:       %[[DEQUANT1:.*]] = IE.Dequantize(%[[ADD]])
    // CHECK:       %[[SS2:.*]] = IE.StridedSlice(%[[DEQUANT1]])
    // CHECK-SAME:   {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 32, 17, 4], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 1]}
    // CHECK:       return  %[[SS2]]
}
