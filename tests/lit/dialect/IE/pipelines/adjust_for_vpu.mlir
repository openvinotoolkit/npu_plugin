// RUN: vpux-opt --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --adjust-for-vpu %s | FileCheck %s

module @Test {

// CHECK-LABEL: @AdjustForVPU
func @AdjustForVPU(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x64xf16> {
    %cts = const.Declare tensor<1x16x5xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x5xf16>>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1], pads_begin = [2], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64xf16> -> tensor<1x1x64xf16>

    return %1 : tensor<1x1x64xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x16x1x5xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x5xf16>, [#const.Reshape<[1, 16, 1, 5]>]>

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 16, 1, 64]} : tensor<1x16x64xf16> -> tensor<1x16x1x64xf16>

    // CHECK: [[VAL1:%.*]] = IE.Convolution([[VAL0]], [[CST]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 2],
    // CHECK-SAME:    post_op = {attrs = {}, name = "IE.ReLU"}, strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x64xf16>

    // CHECK: [[VAL2:%.*]] = IE.AffineReshape([[VAL1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>

    // CHECK: return [[VAL2]] : tensor<1x1x64xf16>
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShift
func @ConvertMultiplyToScaleShift(%arg0: tensor<1x8x128xf32>) -> tensor<1x8x128xf32> {
    %0 = const.Declare tensor<1x1x1xf32> = #const.Content<dense<2.0> : tensor<1x1x1xf32>>
    %1 = IE.Multiply(%arg0, %0)
        { auto_broadcast = "NUMPY" } :
        tensor<1x8x128xf32>, tensor<1x1x1xf32> -> tensor<1x8x128xf32>
    return %1 : tensor<1x8x128xf32>

    // CHECK-NOT:   IE.Multiply
    // CHECK-DAG:       %[[CST_0:.*]] = const.Declare tensor<1x8x1x1xf32> = #const.Content<dense<2.000000e+00> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.Broadcast<1 : i64, 8 : i64>]>
    // CHECK:      %[[VAL0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 8, 1, 128]} : tensor<1x8x128xf32> -> tensor<1x8x1x128xf32>
    // CHECK:      %[[VAL1:.*]] = IE.ScaleShift(%[[VAL0]], %[[CST_0]])
    // CHECK-SAME:      {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x8x1x128xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x1x128xf32>
    // CHECK:      %[[VAL2:.*]] = IE.AffineReshape(%[[VAL1]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 8, 128]} : tensor<1x8x1x128xf32> -> tensor<1x8x128xf32>
    // CHECK:       return %[[VAL2]]
}

}
