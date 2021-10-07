// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorQuant
func @PerTensorQuant(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {
        per_axis = {axis = 1}
    } : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x4x3x4x!qElemType>
    return %0 : tensor<1x4x3x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxis
func @PerAxisQuantOtherAxis(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {
        per_axis = {axis = 2}
    } : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x4x!qElemType>
    return %0 : tensor<1x2x6x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantOtherAxisOffsets
func @PerAxisQuantOtherAxisOffsets(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
    } : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<1x2x6x4x!qElemType>
    return %0 : tensor<1x2x6x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType0 = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = type !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxis
func @PerAxisQuantSameAxis(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType2> {
    %0 = IE.Concat(%arg0, %arg1) {
        per_axis = {axis = 1}
    } : tensor<1x2x3x4x!qElemType0>, tensor<1x2x3x4x!qElemType1> -> tensor<1x4x3x4x!qElemType2>
    return %0 : tensor<1x4x3x4x!qElemType2>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

!qElemType0 = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>
!qElemType1 = type !quant.uniform<u8:f16:1, {3.0000000000000000E-1, 4.0000000000000000E-1}>
!qElemType2 = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1, 3.0000000000000000E-1, 4.0000000000000000E-1}>

// CHECK-LABEL: @PerAxisQuantSameAxisOffsets
func @PerAxisQuantSameAxisOffsets(%arg0: tensor<1x2x3x4x!qElemType0>, %arg1: tensor<1x2x3x4x!qElemType1>) -> tensor<1x4x3x4x!qElemType2> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4x!qElemType0>, tensor<1x2x3x4x!qElemType1> -> tensor<1x4x3x4x!qElemType2>
    return %0 : tensor<1x4x3x4x!qElemType2>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}
