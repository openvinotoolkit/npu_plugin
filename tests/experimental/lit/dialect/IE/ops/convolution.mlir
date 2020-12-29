// RUN: vpux-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FuseConvAndBias
func @FuseConvAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = constant dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1 : i32, 1 : i32],
            pads_begin = [1 : i32, 1 : i32],
            pads_end = [1 : i32, 1 : i32],
            dilations = [1 : i32, 1 : i32]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %bias = constant dense<1.0> : tensor<1x16x1x1xf32>
    %1 = IE.Add(%0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %1 : tensor<1x16x300x300xf32>

    // CHECK:       %[[FILTERS:.*]] = constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK:       %[[BIAS:.*]] = constant dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1 : i32, 1 : i32]
    // CHECK-SAME:      pads_begin = [1 : i32, 1 : i32]
    // CHECK-SAME:      pads_end = [1 : i32, 1 : i32]
    // CHECK-SAME:      strides = [1 : i32, 1 : i32]
    // CHECK:       return %[[VAL0]]
}
