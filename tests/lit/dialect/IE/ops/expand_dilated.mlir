// RUN: vpux-opt --canonicalize %s | FileCheck %s

func @ConstantFolding() -> tensor<1x5x28x31xf16> {
    %cst = const.Declare tensor<1x5x10x11xf16> = #const.Content<dense<1.0> : tensor<1x5x10x11xf16>>
    %0 = IE.ExpandDilated(%cst) {dilations = [3, 3]} : tensor<1x5x10x11xf16> -> tensor<1x5x28x31xf16>
    return %0 : tensor<1x5x28x31xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x5x28x31xf16> = #const.Content<dense<1.000000e+00> : tensor<1x5x10x11xf16>, [#const.ExpandDilated<[3, 3]>]
    // CHECK:       return %[[CST]] : tensor<1x5x28x31xf16>
}
