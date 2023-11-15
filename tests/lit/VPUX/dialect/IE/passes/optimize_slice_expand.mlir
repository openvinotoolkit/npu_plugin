//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --optimize-slice-expand %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @OptimizeSliceExpand
module @OptimizeSliceExpand {

func.func @main(%arg0: tensor<1x80x28x28xf16>) -> tensor<1x80x28x27xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 1] [1, 70, 28, 27] : tensor<1x80x28x28xf16> to tensor<1x70x28x27xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x27xf16> -> tensor<1x80x28x27xf16>
    return %1 : tensor<1x80x28x27xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      tensor<1x80x28x28xf16> to tensor<1x80x28x27xf16>
    // CHECK:       return [[VAR0]] : tensor<1x80x28x27xf16>
}

}

// -----

!qElemType0 = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCastExpand
module @OptimizeSliceQuantizeCastExpand {

func.func @main(%arg0: tensor<1x80x28x28x!qElemType0>) -> tensor<1x80x28x28x!qElemType1> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType0> to tensor<1x70x28x28x!qElemType0>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x70x28x28x!qElemType0> -> tensor<1x70x28x28x!qElemType1>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x28x!qElemType1> -> tensor<1x80x28x28x!qElemType1>
    return %2 : tensor<1x80x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.QuantizeCast(%arg0)
    // CHECK-SAME:      tensor<1x80x28x28x!qElemType0> -> tensor<1x80x28x28x!qElemType1>
    // CHECK:       return [[VAR0]] : tensor<1x80x28x28x!qElemType1>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCastTwoBranchesExpand
module @OptimizeSliceQuantizeCastTwoBranchesExpand {

func.func @main(%arg0: tensor<1x80x28x28x!qElemType0>) -> (tensor<1x70x28x28x!qElemType0, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType0> to tensor<1x70x28x28x!qElemType0>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x70x28x28x!qElemType0> -> tensor<1x70x28x28x!qElemType0, {order = #NHWC}>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x70x28x28x!qElemType0> -> tensor<1x70x28x28x!qElemType1>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x28x!qElemType1> -> tensor<1x80x28x28x!qElemType1>
    return %1, %3 : tensor<1x70x28x28x!qElemType0, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0 
    // CHECK-SAME:  [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType0> to tensor<1x70x28x28x!qElemType0>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]])
    // CHECK-SAME:  {dstOrder = #NHWC} : tensor<1x70x28x28x!qElemType0> -> tensor<1x70x28x28x!qElemType0, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.QuantizeCast(%arg0)
    // CHECK-SAME:      tensor<1x80x28x28x!qElemType0> -> tensor<1x80x28x28x!qElemType1>
    // CHECK:       return [[VAR1]], [[VAR2]] : tensor<1x70x28x28x!qElemType0, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>
}

}

// -----

!qElemType0 = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCast4ChannelExpand
module @OptimizeSliceQuantizeCast4ChannelExpand {

func.func @main(%arg0: tensor<1x16x28x28x!qElemType0>) -> tensor<1x4x28x28x!qElemType1> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 28, 28] : tensor<1x16x28x28x!qElemType0> to tensor<1x1x28x28x!qElemType0>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x1x28x28x!qElemType0> -> tensor<1x1x28x28x!qElemType1>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x1x28x28x!qElemType1> -> tensor<1x4x28x28x!qElemType1>
    return %2 : tensor<1x4x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      tensor<1x16x28x28x!qElemType0> to tensor<1x4x28x28x!qElemType0>
    // CHECK:       [[VAR1:%.+]] = IE.QuantizeCast([[VAR0]])
    // CHECK-SAME:      tensor<1x4x28x28x!qElemType0> -> tensor<1x4x28x28x!qElemType1>
    // CHECK:       return [[VAR1]] : tensor<1x4x28x28x!qElemType1>
}

}

// -----


// CHECK-LABEL: @OptimizeSliceConcatExpand
module @OptimizeSliceConcatExpand {

func.func @main(%arg0: tensor<1x80x4x4xf16>, %arg1: tensor<1x80x4x24xf16>) -> tensor<1x80x4x28xf16> {
   
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x80x4x4xf16> to tensor<1x70x4x4xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x70x4x4xf16>, tensor<1x70x4x24xf16> -> tensor<1x70x4x28xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x28xf16> -> tensor<1x80x4x28xf16>
   return %3 : tensor<1x80x4x28xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
   // CHECK-SAME:      tensor<1x80x4x4xf16>, tensor<1x80x4x24xf16> -> tensor<1x80x4x28xf16>
   // CHECK:       return [[VAR0]] : tensor<1x80x4x28xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceConcatExpand
module @NoOptimizeSliceConcatExpand {

func.func @main(%arg0: tensor<1x80x4x24xf16>, %arg1: tensor<1x80x4x24xf16>) -> tensor<1x144x4x24xf16> {
   
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x70x4x24xf16>, tensor<1x70x4x24xf16> -> tensor<1x140x4x24xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x140x4x24xf16> -> tensor<1x144x4x24xf16>
   return %3 : tensor<1x144x4x24xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x4x24xf16>, tensor<1x70x4x24xf16> -> tensor<1x140x4x24xf16>
   // CHECK-NEXT:  IE.Expand
   // CHECK-SAME:      tensor<1x140x4x24xf16> -> tensor<1x144x4x24xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceConcatAxisHExpand
module @NoOptimizeSliceConcatAxisHExpand {

func.func @main(%arg0: tensor<1x70x20x24xf16>, %arg1: tensor<1x70x20x24xf16>) -> tensor<1x80x20x24xf16> {
   
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x20x24xf16> -> tensor<1x80x20x24xf16>
   return %3 : tensor<1x80x20x24xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   // CHECK-NEXT:  IE.Expand
   // CHECK-SAME:      tensor<1x70x20x24xf16> -> tensor<1x80x20x24xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceConcatAxisHExpand2
module @NoOptimizeSliceConcatAxisHExpand2 {

func.func @main(%arg0: tensor<1x80x20x24xf16>, %arg1: tensor<1x80x20x24xf16>) -> tensor<1x70x30x24xf16> {
   
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 10, 0]} : tensor<1x70x20x24xf16> -> tensor<1x70x30x24xf16>
   return %3 : tensor<1x70x30x24xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   // CHECK-NEXT:  IE.Expand
   // CHECK-SAME:      tensor<1x70x20x24xf16> -> tensor<1x70x30x24xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceExpand
module @NoOptimizeSliceExpand {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x3x4xf16> {
   
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 3, 4] : tensor<1x70x4x4xf16> to tensor<1x70x3x4xf16> 
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   return %1 : tensor<1x80x3x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x70x4x4xf16> to tensor<1x70x3x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]])
   // CHECK-SAME:      tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   // CHECK:       return [[VAR1]] : tensor<1x80x3x4xf16>

}
}

// -----

// CHECK-LABEL: @NotOptimizeSliceExpandDueToOffset
module @NotOptimizeSliceExpandDueToOffset {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x20x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 60, 0, 0] [1, 10, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x10x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x10x4x4xf16> -> tensor<1x20x4x4xf16>
   return %1 : tensor<1x20x4x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x70x4x4xf16> to tensor<1x10x4x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]])
   // CHECK-SAME:      tensor<1x10x4x4xf16> -> tensor<1x20x4x4xf16>
   // CHECK:       return [[VAR1]] : tensor<1x20x4x4xf16>
}
}

// -----

// CHECK-LABEL: @DeleteSliceExpand
module @DeleteSliceExpand {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x4x4xf16> {
   
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   return %1 : tensor<1x80x4x4xf16>

   // CHECK-NOT:   IE.Slice
   // CHECK:       [[VAR0:%.+]] = IE.Expand(%arg0)
   // CHECK-SAME:      tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
   // CHECK:       return [[VAR0]] : tensor<1x80x4x4xf16>

}
}

// -----

// CHECK-LABEL: @NoSliceExpand
module @NoSliceExpand {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x4x4xf16> {
   
   %0 = IE.Slice %arg0 [0, 10, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   return %1 : tensor<1x80x4x4xf16>

   // CHECK:       [[SLICE:%.*]] = IE.Slice %arg0 [0, 10, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   // CHECK:       [[EXPAND:%.*]] = IE.Expand([[SLICE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   // CHECK:       return [[EXPAND]] : tensor<1x80x4x4xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeTwoBranchesSliceExpand
module @OptimizeTwoBranchesSliceExpand {

func.func @main(%arg0: tensor<1x80x4x4xf16>) -> (tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>) {


   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 3, 4] : tensor<1x80x4x4xf16> to tensor<1x70x3x4xf16> 
   %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x70x3x4xf16> -> tensor<1x70x3x4xf16, {order = #NHWC}>
   %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   return %1, %2 : tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x80x4x4xf16> to tensor<1x70x3x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]])
   // CHECK-SAME:      tensor<1x70x3x4xf16> -> tensor<1x70x3x4xf16, {order = #NHWC}>
   // CHECK:       [[VAR2:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x80x4x4xf16> to tensor<1x80x3x4xf16>
   // CHECK:       return [[VAR1]], [[VAR2]] : tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>
}
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePattern
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @OptimizeExpandSlicePattern(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x3x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   return %1 : tensor<1x3x32x32xf16>

   // CHECK-NOT:    IE.Expand
   // CHECK-NOT:    IE.Slice
   // CHECK:        return [[INPUT]] : tensor<1x3x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternUnsupportedOffset
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @OptimizeExpandSlicePatternUnsupportedOffset(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x3x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   return %1 : tensor<1x3x32x32xf16>

   // Nothing should be changed
   // The input data of the Expand and the output data of the Slice are different because of the offset
   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 3, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   // CHECK:        [[SLICE]] : tensor<1x3x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternToExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @OptimizeExpandSlicePatternToExpand(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x4x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   return %1 : tensor<1x4x32x32xf16>

   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x4x32x32xf16>
   // CHECK-NOT:    IE.Slice 
   // CHECK:        [[EXPAND]] : tensor<1x4x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternToSlice
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x5x32x32xf16>
func.func @OptimizeExpandSlicePatternToSlice(%arg0: tensor<1x5x32x32xf16>) -> tensor<1x4x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   return %1 : tensor<1x4x32x32xf16>

   // CHECK-NOT:    IE.Expand
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x5x32x32xf16> to tensor<1x4x32x32xf16> 
   // CHECK:        [[SLICE]] : tensor<1x4x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceHSwishExpand
module @OptimizeSliceHSwishExpand {

func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.HSwish(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.HSwish(%arg0)
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSwishExpand
module @OptimizeSliceSwishExpand {

func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.Swish(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.Swish(%arg0)
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

// CHECK-LABEL: @OptimizeExpandOverSameDimWithSingleSlice
module @OptimizeExpandOverSameDimWithSingleSlice {

func.func @main(%arg0: tensor<1x96x180x320xf16>, %arg1: tensor<1x96x180x320xf16>) -> tensor<1x192x180x320xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 84, 180, 320] : tensor<1x96x180x320xf16> to tensor<1x84x180x320xf16>
   %1 = IE.Concat(%arg1, %0) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<1x96x180x320xf16>, tensor<1x84x180x320xf16> -> tensor<1x180x180x320xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x180x180x320xf16> -> tensor<1x192x180x320xf16>
   return %2 : tensor<1x192x180x320xf16>

   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x96x180x320xf16>, tensor<1x96x180x320xf16> -> tensor<1x192x180x320xf16>

}
}
