// RUN: vpux-opt --split-input-file --split-fake-quant %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @SingleQuantParams
func @SingleQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.0> : tensor<1x1x1x1xf32>>
    %input_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<255.0> : tensor<1x1x1x1xf32>>
    %output_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.0> : tensor<1x1x1x1xf32>>
    %output_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<255.0> : tensor<1x1x1x1xf32>>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
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

!qElemType0 = type !quant.uniform<u8:f32, 0.078431372549019607:128>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @DifferentQuantParams
func @DifferentQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<-10.0> : tensor<1x1x1x1xf32>>
    %input_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<10.0> : tensor<1x1x1x1xf32>>
    %output_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.0> : tensor<1x1x1x1xf32>>
    %output_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<255.0> : tensor<1x1x1x1xf32>>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
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

!qElemType0 = type !quant.uniform<u8:f32, 0.039215686274509803>
!qElemType1 = type !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @OneDifferentQuantParam
func @OneDifferentQuantParam(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.0> : tensor<1x1x1x1xf32>>
    %input_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<10.0> : tensor<1x1x1x1xf32>>
    %output_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.0> : tensor<1x1x1x1xf32>>
    %output_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<255.0> : tensor<1x1x1x1xf32>>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
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

!qElemType = type !quant.uniform<u8:f32:1, {0.011764705882352941:85,0.015686274509803921:64}>

// CHECK-LABEL: @BroadcastQuantParam
func @BroadcastQuantParam(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<-1.0> : tensor<1x1x1x1xf32>>
    %input_high = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>>
    %output_low = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>>
    %output_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<2.0> : tensor<1x1x1x1xf32>>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
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

!qElemType = type !quant.uniform<u8:f32:1, {0.011764705882352941:85,0.015686274509803921:64}>

// CHECK-LABEL: @BroadcastQuantParamDiffRanks
func @BroadcastQuantParamDiffRanks(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1xf32> = #const.Content<dense<-1.0> : tensor<1xf32>>
    %input_high = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>>
    %output_low = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>>
    %output_high = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
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

!qElemType = type !quant.uniform<i8:f32, 0.078431372549019607>

// CHECK-LABEL: @UseDequantize
func @UseDequantize() -> tensor<1x3x30x30xf32> {
    %input = const.Declare tensor<1x3x30x30xf32> =
        #const.Content<dense<10> : tensor<1x3x30x30xui8>, [#const.ConvertElemType<f32>]>

    %input_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<-10.0> : tensor<1x1x1x1xf32>>
    %input_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<10.0> : tensor<1x1x1x1xf32>>
    %output_low = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.0> : tensor<1x1x1x1xf32>>
    %output_high = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<255.0> : tensor<1x1x1x1xf32>>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<1x3x30x30x!qElemType> =
    // CHECK-SAME:      #const.Content<dense<10> : tensor<1x3x30x30xui8>
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
func @UseRescale() -> tensor<1x2x30x30xf32> {
    %input = const.Declare tensor<1x2x30x30xf32> = #const.Content<dense<1.0> : tensor<1x2x30x30xf32>>
    %input_low = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[-2.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>>
    %input_high = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[2.0]],[[1.0]]]]> : tensor<1x2x1x1xf32>>
    %output_low = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[-1.0]],[[-0.5]]]]> : tensor<1x2x1x1xf32>>
    %output_high = const.Declare tensor<1x2x1x1xf32> = #const.Content<dense<[[[[1.0]],[[0.5]]]]> : tensor<1x2x1x1xf32>>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<1x2x30x30xf32>>

    // CHECK:       [[VAL1:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<1x2x30x30xf32>, [#const.Rescale<2.000000e+00 : f64>]>

    // CHECK:       return [[VAL1]]
}
