// RUN: vpux-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @Eliminate
func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Unsqueeze(%arg0) { axes_value = [] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Unsqueeze
    // CHECK:     return %arg0
}

// CHECK-LABEL: @ConstFold
func @ConstFold() -> tensor<1x1x4x4xf32> {
    %0 = const.Declare tensor<4x4xf32> = #const.Content<dense<1.0> : tensor<4x4xf32>>
    %1 = IE.Unsqueeze(%0) { axes_value = [0, 1] } : tensor<4x4xf32> -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>

    // CHECK:       [[VAL0:%.+]] = const.Declare tensor<1x1x4x4xf32> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<4x4xf32>, [#const.Reshape<[1, 1, 4, 4]>]>
    // CHECK-NOT:   IE.Unsqueeze
    // CHECK:       return [[VAL0]]
}

// CHECK-LABEL: @FuseWithReshape
func @FuseWithReshape(%arg0: tensor<16xf32>) -> tensor<4x1x4x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    %1 = IE.Unsqueeze(%0) { axes_value = [1, 3] } : tensor<4x4xf32> -> tensor<4x1x4x1xf32>
    return %1 : tensor<4x1x4x1xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [4, 1, 4, 1]} : tensor<16xf32> -> tensor<4x1x4x1xf32>
    // CHECK: return [[VAL0]] : tensor<4x1x4x1xf32>
}

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<4x4xf32>) -> tensor<4x1x4x1xf32> {
    %0 = const.Declare tensor<2xsi64> = #const.Content<dense<[1, 3]> : tensor<2xsi64>>
    %1 = IE.Unsqueeze(%arg0, %0) : tensor<4x4xf32>, tensor<2xsi64> -> tensor<4x1x4x1xf32>
    return %1 : tensor<4x1x4x1xf32>

    // CHECK: [[VAL0:%.+]] = IE.Unsqueeze(%arg0) {axes_value = [1, 3]} : tensor<4x4xf32> -> tensor<4x1x4x1xf32>
    // CHECK: return [[VAL0]] : tensor<4x1x4x1xf32>
}
