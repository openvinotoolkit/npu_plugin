// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

func @FoldExpand(%arg0: tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4x4xf16> -> tensor<1x8x4x4xf16>
  return %0 : tensor<1x8x4x4xf16>

  // CHECK-NOT: IE.Expand
}
