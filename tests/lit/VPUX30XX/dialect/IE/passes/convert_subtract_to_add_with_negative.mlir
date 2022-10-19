// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-subtract-to-negative-add %s | FileCheck %s

// CHECK-LABEL: @ConvertSubtractToAddWithNegative
func @ConvertSubtractToAddWithNegative(%arg0: tensor<1x16x32xf32>) -> tensor<1x16x32xf32> {
    %cst = const.Declare tensor<1x16x1xf32> = #const.Content<dense<2.0> : tensor<1x16x1xf32>>
    %0 = IE.Subtract(%arg0, %cst)
        { auto_broadcast = "NUMPY" } :
        tensor<1x16x32xf32>, tensor<1x16x1xf32> -> tensor<1x16x32xf32>

    return %0 : tensor<1x16x32xf32>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x16x1xf32> = #const.Content<dense<2.000000e+00> : tensor<1x16x1xf32>>
    // CHECK:       %[[VAL0:.*]] = IE.Negative(%[[CST]]) : tensor<1x16x1xf32> -> tensor<1x16x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.Add(%arg0, %[[VAL0]]) {auto_broadcast = "NUMPY"} : tensor<1x16x32xf32>, tensor<1x16x1xf32> -> tensor<1x16x32xf32>
    // CHECK:       return %[[VAL1]]
}
