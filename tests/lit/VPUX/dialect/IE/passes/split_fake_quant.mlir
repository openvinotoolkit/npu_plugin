//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @SingleQuantParams
func.func @SingleQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType0 = !quant.uniform<u8:f32, 0.078431372549019607:128>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @DifferentQuantParams
func.func @DifferentQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType0}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

!qElemType0 = !quant.uniform<u8:f32, 0.039215686274509803>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @OneDifferentQuantParam
func.func @OneDifferentQuantParam(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType0}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f32:1, {0.011764705882352941:85,0.015686274509803921:64}>

// CHECK-LABEL: @BroadcastQuantParam
func.func @BroadcastQuantParam(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x2x30x30xf32> ->
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x2x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f32:1, {0.011764705882352941:85,0.015686274509803921:64}>

// CHECK-LABEL: @BroadcastQuantParamDiffRanks
func.func @BroadcastQuantParamDiffRanks(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1xf32> = dense<-1.0> : tensor<1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x2x30x30xf32> ->
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x2x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f32, 0.078431372549019607>

// CHECK-LABEL: @UseDequantize
func.func @UseDequantize() -> tensor<1x3x30x30xf32> {
    %input = const.Declare tensor<1x3x30x30xf32> =
        dense<10> : tensor<1x3x30x30xui8>, [#const.ConvertElemType<f32>]

    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1x3x30x30x!qElemType> =
    // CHECK-SAME:      dense<10> : tensor<1x3x30x30xui8
    // CHECK-SAME:      #const.ConvertElemType<f32>
    // CHECK-SAME:      #const.ConvertElemType<si8>
    // CHECK-SAME:      #const.QuantCast<!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>
    // CHECK-SAME:      -> tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @UseRescale
func.func @UseRescale() -> tensor<1x2x30x30xf32> {
    %input = const.Declare tensor<1x2x30x30xf32> = dense<1.0> : tensor<1x2x30x30xf32>
    %input_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-2.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[1.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-0.5]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[1.0]],[[0.5]]]]> : tensor<1x2x1x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x30x30xf32>

    // CHECK-DAG:       [[VAL1:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x30x30xf32>, [#const.Rescale<2.000000e+00 : f64>]

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8<0:1>:f32, 1.000000e+00>

// CHECK-LABEL: @Level2Quantization
func.func @Level2Quantization(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 2 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @Scalar0AsInputRange
func.func @Scalar0AsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

!qElemType0 = !quant.uniform<u8:f32, 7.000000e+00>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @Scalar7AsInputRange
func.func @Scalar7AsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<7.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<7.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>


    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType0}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

!qElemType0 = !quant.uniform<u8:f32, 4.000000e+00>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NegativeScalarAsInputRange
func.func @NegativeScalarAsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-4.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<-4.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>


    // CHECK:       [[VAL0:%.*]] = IE.Quantize(%arg0)
    // CHECK-SAME:      {dstElemType = !qElemType0}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType0> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK-LABEL: @PerChannelQuant
func.func @PerChannelQuant(%arg0: tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32> {
    %input_low = const.Declare tensor<1x16x1x1xf32> = 
        dense<[[[[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[-3.750000e-01]], 
        [[0.000000e+00]], [[-0.297849566]], [[-0.382785916]], [[-0.399385154]], [[-3.750000e-01]], 
        [[0.000000e+00]], [[-0.381789744]], [[0.000000e+00]], [[-3.750000e-01]], [[-3.750000e-01]], [[-0.389199734]]]]> : tensor<1x16x1x1xf32>
    %input_high = const.Declare tensor<1x16x1x1xf32> = dense<[[[[2.88798332]], [[17.5819988]], [[23.3847122]],
        [[6.1077733]], [[1.875000e+01]], [[3.81057024]], [[0.192161009]], [[13.5615578]], [[12.3310165]],
        [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]], [[2.70967746]], [[8.63315773]]]]> : tensor<1x16x1x1xf32>
    %output_low = const.Declare tensor<1x16x1x1xf32> = dense<[[[[0.000000e+00]], [[0.000000e+00]],
        [[0.000000e+00]], [[0.000000e+00]], [[-3.750000e-01]], [[0.000000e+00]], [[-0.297849566]],
        [[-0.382785916]], [[-0.399385154]], [[-3.750000e-01]], [[0.000000e+00]], [[-0.381789744]],
        [[0.000000e+00]], [[-3.750000e-01]], [[-3.750000e-01]], [[-0.389199734]]]]> : tensor<1x16x1x1xf32>
    %output_high =  const.Declare tensor<1x16x1x1xf32> = dense<[[[[2.88798332]], [[17.5819988]],
        [[23.3847122]], [[6.1077733]], [[1.875000e+01]], [[3.81057024]], [[0.192161009]], [[13.5615578]],
        [[12.3310165]], [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]],
        [[2.70967746]], [[8.63315773]]]]> : tensor<1x16x1x1xf32>

    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high) 
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : 
        tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x112x112xf32>

    return %fq : tensor<1x16x112x112xf32>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Deuantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----

// CHECK-LABEL: @PerChannelQuantInput
func.func @PerChannelQuantInput(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299xf16> {
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<2.63867188> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.65820313> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[2.551250e+02]], [[2.670000e+02]], [[2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_low =  const.Declare tensor<1x3x1x1xf16> = dense<[[[[-49.28125]], [[-35.65625]], [[-31.828125]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high) 
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x299x299xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x299x299xf16>

    return %1 : tensor<1x3x299x299xf16>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Deuantize
    // CHECK-NOT:       IE.QuantizeCast
}
