// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

!qElemType = type !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @ConcatLargeOffsetStride
func @ConcatLargeOffsetStride(%arg0: tensor<1x2x3x4x!qElemType>, %arg1: tensor<1x2x3x4x!qElemType>) -> tensor<2x2x3x4x!qElemType> {
    %0 = IE.Concat(%arg0, %arg1) {
        per_axis = {axis = 0, offset = 1, stride = 2}
    } : tensor<1x2x3x4x!qElemType>, tensor<1x2x3x4x!qElemType> -> tensor<2x2x3x4x!qElemType>
    return %0 : tensor<2x2x3x4x!qElemType>

    // The operation should be parsed and verified successfully
    // CHECK: IE.Concat
}

// -----

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

// -----

// CHECK-LABEL: @ConvertPerAxisToOffsets
func @ConvertPerAxisToOffsets(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {
        per_axis = {axis = 1}
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    return %0: tensor<1x4x3x4xf32>

    // CHECK:     [[VAL_0:%.*]] = IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]}
    // CHECK-SAME:     tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    // CHECK:     return [[VAL_0]] : tensor<1x4x3x4xf32>
}

// -----

// CHECK-LABEL: @FuseConcatWithOffsetsAndOtherOp
func @FuseConcatWithOffsetsAndOtherOp(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>,
                                      %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>,
                                      %arg4: tensor<1x2x4x3xf32>) -> tensor<1x10x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %1 = IE.Concat(%arg2, %arg3) {
        per_axis = {axis = 1}
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %2 = IE.Reshape(%arg4) { shape_value = [1, 2, 3, 4] } : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    %3 = IE.Concat(%0, %1, %2) {
        static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0]]
    } : tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    return %3: tensor<1x10x3x4xf32>

    // CHECK-DAG:     [[RES_0:%.*]] = IE.Reshape(%arg4) {shape_value = [1, 2, 3, 4]} : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    // CHECK:     [[VAL_0:%.*]] = IE.Concat(%arg0, %arg1, %arg2, %arg3, [[RES_0]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0]]}
    // CHECK-SAME:     tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    // CHECK:     return [[VAL_0]] : tensor<1x10x3x4xf32>
}

// -----

// CHECK-LABEL: @FuseConcatWithPerAxis
func @FuseConcatWithPerAxis(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>,
                            %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>,
                            %arg4: tensor<1x2x4x3xf32>) -> tensor<1x10x3x4xf32> {
    %0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %1 = IE.Concat(%arg2, %arg3) {
        per_axis = {axis = 1}
    } : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x4x3x4xf32>
    %2 = IE.Reshape(%arg4) { shape_value = [1, 2, 3, 4] } : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    %3 = IE.Concat(%0, %1, %2) {
        per_axis = {axis = 1}
    } : tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    return %3 : tensor<1x10x3x4xf32>

    // CHECK-DAG:     [[RES_0:%.*]] = IE.Reshape(%arg4) {shape_value = [1, 2, 3, 4]} : tensor<1x2x4x3xf32> -> tensor<1x2x3x4xf32>
    // CHECK:     [[VAL_0:%.*]] = IE.Concat(%arg0, %arg1, %arg2, %arg3, [[RES_0]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0]]}
    // CHECK-SAME:     tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32> -> tensor<1x10x3x4xf32>
    // CHECK:     return [[VAL_0]] : tensor<1x10x3x4xf32>
}

// -----

// CHECK-LABEL: @OneInputFold
func @OneInputFold(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Concat(%arg0) { per_axis = {axis = 1} } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Concat
    // CHECK:     return %arg0
}
