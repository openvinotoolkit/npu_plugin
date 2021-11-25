// RUN: vpux-opt --split-input-file --convert-layers-to-EMU %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
func @SingleLayer(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %0: tensor<1x1x1x1000xf16>

    // CHECK: [[VAR0:%.*]] = EMU.SoftMaxUPA {axisInd = 3 : i64} inputs(%arg0 : tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
}

// -----

// CHECK-LABEL: @ConstantLayer
func @ConstantLayer(%arg0: tensor<1x2x2x2xf16>) -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> = #const.Content<dense<1.0> : tensor<1x2x2x2xf16>>
    return %0: tensor<1x2x2x2xf16>

    // CHECK:       [[VAR0:%.*]] = const.Declare tensor<1x2x2x2xf16>
    // CHECK-SAME:      = #const.Content<dense<1.000000e+00> : tensor<1x2x2x2xf16>>

    // CHECK: return [[VAR0]] : tensor<1x2x2x2xf16>
}

// -----

// CHECK-LABEL: @ReshapeInGraph
func @ReshapeInGraph(%arg0: tensor<1x512xf16>) -> tensor<1x512xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 512, 1, 1]}: tensor<1x512xf16> -> tensor<1x512x1x1xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 512]}: tensor<1x512x1x1xf16> -> tensor<1x512xf16>
    return %2: tensor<1x512xf16>

    // CHECK:       [[VAR0:%.*]] = EMU.Reshape inputs(%arg0 : tensor<1x512xf16>) -> tensor<1x512x1x1xf16>
    // CHECK:       [[VAR1:%.*]] = EMU.SoftMaxUPA {axisInd = 1 : i64} inputs(%0 : tensor<1x512x1x1xf16>) -> tensor<1x512x1x1xf16>
    // CHECK:       [[VAR2:%.*]] = EMU.Reshape inputs([[VAR1]] : tensor<1x512x1x1xf16>) -> tensor<1x512xf16>

    // CHECK: return [[VAR2]] : tensor<1x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteCast
func @PermuteCast(%arg0: tensor<1x12x16x16xf16,  {order = #NHWC}>) -> tensor<1x16x16x12xf16> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW}
        : tensor<1x12x16x16xf16,  {order = #NHWC}> -> tensor<1x16x16x12xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1}
        : tensor<1x16x16x12xf16> -> tensor<1x16x16x12xf16>
    return %1 : tensor<1x16x16x12xf16>

    //CHECK:        [[VAR0:%.*]] = EMU.PermuteUPA {order_value = #NCHW}
    //CHECK:        [[VAR1:%.*]] = EMU.SoftMaxUPA {axisInd = 1 : i64}
    //CHECK-SAME:       inputs([[VAR0]] : tensor<1x16x16x12xf16>)
    //CHECK-SAME:       -> tensor<1x16x16x12xf16>
    //CHECK:        return [[VAR1]] : tensor<1x16x16x12xf16>
}
