// RUN: vpux-opt --canonicalize %s | FileCheck %s

func @FoldExpand(%arg0: tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK: return %arg0 : tensor<1x8x4x4xf16>
}

func @FoldSliceExpand(%arg0: tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 72, 56, 56] : tensor<1x80x56x56xf16> to tensor<1x72x56x56xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x72x56x56xf16> -> tensor<1x80x56x56xf16>
    return %1 : tensor<1x80x56x56xf16>

    // CHECK: return %arg0 : tensor<1x80x56x56xf16>
}

func @ConstantFolding() -> tensor<1x11x12x12xf16> {
    %cst = const.Declare tensor<1x5x10x11xf16> = #const.Content<dense<1.0> : tensor<1x5x10x11xf16>>
    %0 = IE.Expand(%cst) {pads_begin = [0, 3, 0, 1], pads_end = [0, 3, 2, 0]} : tensor<1x5x10x11xf16> -> tensor<1x11x12x12xf16>
    return %0 : tensor<1x11x12x12xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x11x12x12xf16> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<1x5x10x11xf16>, [#const.PadWithZero<[0, 3, 0, 1], [0, 3, 2, 0]>]>
    // CHECK:       return %[[CST]] : tensor<1x11x12x12xf16>
}
