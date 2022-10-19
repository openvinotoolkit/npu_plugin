// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shuffle-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertShuffleChannels
func @ConvertShuffleChannels(%arg0: tensor<1x4x3x2xf16>) -> tensor<1x4x3x2xf16> {

  %prob = IE.ShuffleChannels(%arg0) {axis = 1, group = 2} : tensor<1x4x3x2xf16> -> tensor<1x4x3x2xf16>

  return %prob : tensor<1x4x3x2xf16>

  //CHECK:              [[VAL0:%.*]] = IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}:                  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 2, 6]} : tensor<1x4x3x2xf16> -> tensor<1x2x2x6xf16>
  //CHECK:              [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NHCW} : tensor<1x2x2x6xf16> -> tensor<1x2x2x6xf16>
  //CHECK:              [[VAL2:%.*]] = IE.AffineReshape([[VAL1]])
  //CHECK-SAME{LITERAL}:                  {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4, 3, 2]} : tensor<1x2x2x6xf16> -> tensor<1x4x3x2xf16>
  //CHECK:              return [[VAL2]]
}
