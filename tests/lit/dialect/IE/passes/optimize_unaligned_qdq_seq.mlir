// RUN: vpux-opt --split-input-file --optimize-unaligned-qdq-seq %s | FileCheck %s

!qElemType1 = type !quant.uniform<u8:f16, 2.4627450980392158>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeQuantDequantSequence
func @OptimizeQuantDequantSequence(%arg0 : tensor<1x48x1x1xf16>, %arg1 : tensor<512x48x1x1xf16>) -> tensor<1x128x1x8xf16, {order = #NHWC}> {
  %cst_1 = const.Declare tensor<128x64x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.0> : tensor<128x64x1x1xf16, {order = #NHWC}>>
  %cst_2 = const.Declare tensor<1x128x1x1xf16> = #const.Content<dense<1.0> : tensor<1x128x1x1xf16>>
  %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x1x1xf16>, tensor<512x48x1x1xf16> -> tensor<1x512x1x1xf16, {order = #NHWC}>
  %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>

  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x512xf16> -> tensor<1x16x1x512xf16>
  %4 = IE.MemPermute(%3) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x1x512xf16> -> tensor<1x16x1x512xf16, {order = #NHWC}>
  %5 = IE.And(%4, %4) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x1x512xf16, {order = #NHWC}>, tensor<1x16x1x512xf16, {order = #NHWC}> -> tensor<1x16x1x512x!qElemType1, {order = #NHWC}>
  %6 = IE.And(%5, %5) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x16x1x512x!qElemType1, {order = #NHWC}>, tensor<1x16x1x512x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x512xf16, {order = #NHWC}>
  %7 = IE.Slice %6 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x16x1x512xf16, {order = #NHWC}> to tensor<1x1x1x512xf16, {order = #NHWC}>
  %8 = IE.MemPermute(%7) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x1x1x512xf16>


  %9 = IE.AffineReshape(%8) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 8, 64]} : tensor<1x1x1x512xf16> -> tensor<1x1x8x64xf16>
  %10 = IE.PermuteCast(%9) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x8x64xf16> -> tensor<1x64x1x8xf16, {order = #NHWC}>
  %11 = IE.Convolution(%10, %cst_1, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x8xf16, {order = #NHWC}>, tensor<128x64x1x1xf16, {order = #NHWC}>, tensor<1x128x1x1xf16> -> tensor<1x128x1x8xf16, {order = #NHWC}>
  return %11 :tensor<1x128x1x8xf16, {order = #NHWC}>

  // CHECK:  [[VAL1:%.*]] = IE.Convolution(%arg0, %arg1)
  // CHECK:  [[VAL2:%.*]] = IE.And([[VAL1]], [[VAL1]])
  // CHECK-SAME: -> tensor<1x512x1x1x!qElemType, {order = #NHWC}>
  // CHECK:  [[VAL3:%.*]] = IE.And([[VAL2]], [[VAL2]])
  // CHECK-SAME: -> tensor<1x512x1x1xf16, {order = #NHWC}>
  // CHECK:  [[VAL4:%.*]] = IE.MemPermute([[VAL3]])
  // CHECK:  [[VAL5:%.*]] = IE.AffineReshape([[VAL4]])
  // CHECK:  IE.AffineReshape([[VAL5]])

}
