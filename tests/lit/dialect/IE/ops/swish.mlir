// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %0 = const.Declare tensor<1x16xf32> = #const.Content<dense<1.0> : tensor<1x16xf32>>
    %1 = IE.Swish(%arg0, %0) :
        tensor<1x16x300x300xf32>, tensor<1x16xf32> -> tensor<1x16x300x300xf32>
    return %1 : tensor<1x16x300x300xf32>

    // CHECK: %[[VAL0:.*]] = IE.Swish(%arg0) {beta_value = 1.000000e+00 : f32} : tensor<1x16x300x300xf32> -> tensor<1x16x300x300xf32>
    // CHECK: return %[[VAL0]]
}
