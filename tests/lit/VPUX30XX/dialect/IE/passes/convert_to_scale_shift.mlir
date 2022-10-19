// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-to-scale-shift %s | FileCheck %s

// CHECK-LABEL: @ConvertAddToScaleShift
func @ConvertAddToScaleShift(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x3x1x1xf32> = #const.Content<dense<2.0> : tensor<1x3x1x1xf32>>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = #const.Content<dense<2.000000e+00> : tensor<1x3x1x1xf32>>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftBroadcastChannels
func @ConvertAddToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<2.0> : tensor<1x1x1x1xf32>>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = #const.Content<dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 3 : i64>]>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftReversedInputs
func @ConvertAddToScaleShiftReversedInputs(%arg0: tensor<1x3x1x1xf32>, %arg1: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %0 = IE.Add(%arg0, %arg1)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x1x1xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg1, %arg0) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShift
func @ConvertMultiplyToScaleShift(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = #const.Content<dense<3.0> : tensor<1x3x1x1xf32>>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = #const.Content<dense<3.000000e+00> : tensor<1x3x1x1xf32>>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]]) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftBroadcastChannels
func @ConvertMultiplyToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<3.0> : tensor<1x1x1x1xf32>>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = #const.Content<dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 3 : i64>]>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]]) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftReversedInputs
func @ConvertMultiplyToScaleShiftReversedInputs(%arg0: tensor<1x3x1x1xf32>, %arg1: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x1x1xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg1, %arg0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}
