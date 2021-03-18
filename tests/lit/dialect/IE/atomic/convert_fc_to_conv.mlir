// RUN: vpux-opt --split-input-file --convert-fc-to-conv %s | FileCheck %s

// CHECK-LABEL: @ConvertFullyConnectedToConvolution
func @ConvertFullyConnectedToConvolution(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = IE.Constant tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = IE.Constant tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK:       %[[WEIGHTS:.*]] = IE.Constant tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK:       %[[BIAS:.*]] = IE.Constant tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       %[[SHAPE0:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 16, 1, 1]> : tensor<4xsi64>
    // CHECK:       %[[VAL0:.*]] = IE.Reshape(%arg0, %[[SHAPE0]]) : tensor<1x16xf32>, tensor<4xsi64> -> tensor<1x16x1x1xf32>
    // CHECK:       %[[SHAPE1:.*]] = IE.Constant tensor<4xsi64> = dense<[64, 16, 1, 1]> : tensor<4xsi64>
    // CHECK:       %[[VAL1:.*]] = IE.Reshape(%[[WEIGHTS]], %[[SHAPE1]]) : tensor<64x16xf32>, tensor<4xsi64> -> tensor<64x16x1x1xf32>
    // CHECK:       %[[SHAPE2:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 64, 1, 1]> : tensor<4xsi64>
    // CHECK:       %[[VAL2:.*]] = IE.Reshape(%[[BIAS]], %[[SHAPE2]]) : tensor<1x64xf32>, tensor<4xsi64> -> tensor<1x64x1x1xf32>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[VAL0]], %[[VAL1]], %[[VAL2]])
    // CHECK-SAME:      dilations = [1 : i32, 1 : i32]
    // CHECK-SAME:      pads_begin = [0 : i32, 0 : i32]
    // CHECK-SAME:      pads_end = [0 : i32, 0 : i32]
    // CHECK-SAME:      strides = [1 : i32, 1 : i32]
    // CHECK:       %[[SHAPE3:.*]] = IE.Constant tensor<2xsi64> = dense<[1, 64]> : tensor<2xsi64>
    // CHECK:       %[[VAL3:.*]] = IE.Reshape(%[[CONV]], %[[SHAPE3]]) : tensor<1x64x1x1xf32>, tensor<2xsi64> -> tensor<1x64xf32>
    // CHECK:       return %[[VAL3]]
}

