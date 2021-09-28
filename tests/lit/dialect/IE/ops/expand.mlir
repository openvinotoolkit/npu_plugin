// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

func @FoldExpand(%arg0: tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4x4xf16> -> tensor<1x8x4x4xf16>
  return %0 : tensor<1x8x4x4xf16>

  // CHECK-NOT: IE.Expand
}

// -----

func @FoldSliceExpand(%arg0: tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16> {

  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 72, 56, 56] : tensor<1x80x56x56xf16> to tensor<1x72x56x56xf16>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x72x56x56xf16> -> tensor<1x80x56x56xf16>
  return %1 : tensor<1x80x56x56xf16>

  // CHECK-NOT: IE.Slice
  // CHECK-NOT: IE.Expand
}
