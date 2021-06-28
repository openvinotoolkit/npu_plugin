// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ConstFold
func @ConstFold() -> tensor<1x8x4x4xf32> {
    %0 = const.Declare tensor<1x8x4x4xf32> = #const.Content<dense<5.0> : tensor<1x8x4x4xf32>>
    %1 = const.Declare tensor<1x8x4x4xf32> = #const.Content<dense<0.0> : tensor<1x8x4x4xf32>>
    %2 = IE.Add(%0, %1)
        { auto_broadcast = "NUMPY" } :
        tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32> -> tensor<1x8x4x4xf32>
    return %2 : tensor<1x8x4x4xf32>

    // CHECK:       %[[VAL0:.*]] = const.Declare tensor<1x8x4x4xf32> = #const.Content<dense<5.000000e+00> : tensor<1x8x4x4xf32>>
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   IE.Add
    // CHECK:       return %[[VAL0]]
}
