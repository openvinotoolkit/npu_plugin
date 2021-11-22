// RUN: vpux-opt --split-input-file --fc-inputs-to-2d %s | FileCheck %s

// CHECK-LABEL: @FullyConnectedInputsTo2d
func @FullyConnectedInputsTo2d(%arg0: tensor<2x1x512xf32>) -> tensor<2x1x40xf32> {
    %cst = const.Declare tensor<2x512x40xf32> = #const.Content<dense<1.0> : tensor<2x512x40xf32>>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x1x512xf32>, tensor<2x512x40xf32> -> tensor<2x1x40xf32>

    return %0 : tensor<2x1x40xf32>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<2x512x40xf32> = #const.Content<dense<1.000000e+00> : tensor<2x512x40xf32>>
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.Reshape(%[[IN_1]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [1, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.Reshape(%[[IN_2]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[CST_1:.*]] = IE.Slice %[[CST]] [0, 0, 0] [1, 512, 40] : tensor<2x512x40xf32> to tensor<1x512x40xf32>
    // CHECK:       %[[CST_1_2D:.*]] = IE.Reshape(%[[CST_1]]) {shape_value = [512, 40]} : tensor<1x512x40xf32> -> tensor<512x40xf32>
    // CHECK:       %[[CST_2:.*]] = IE.Slice %[[CST]] [1, 0, 0] [1, 512, 40] : tensor<2x512x40xf32> to tensor<1x512x40xf32>
    // CHECK:       %[[CST_2_2D:.*]] = IE.Reshape(%[[CST_2]]) {shape_value = [512, 40]} : tensor<1x512x40xf32> -> tensor<512x40xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = {axis = 0 : i64}} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 1, 40]} : tensor<2x40xf32> -> tensor<2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<2x1x40xf32>
}
