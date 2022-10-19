// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Eliminate
func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.Squeeze(%arg0) { axes_value = [] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.Squeeze
    // CHECK:     return %arg0
}

// CHECK-LABEL: @ConstFold
func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<1x1x4x4xf32> = #const.Content<dense<1.0> : tensor<1x1x4x4xf32>>
    %1 = IE.Squeeze(%0) { axes_value = [0, 1] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<1x1x4x4xf32>, [#const.Reshape<[4, 4]>]>
    // CHECK-NOT:   IE.Squeeze
    // CHECK:       return [[VAL0]]
}

// CHECK-LABEL: @FuseWithReshape
func @FuseWithReshape(%arg0: tensor<16x1xf32>) -> tensor<4x4xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 1, 4, 4] } : tensor<16x1xf32> -> tensor<1x1x4x4xf32>
    %1 = IE.Squeeze(%0) { axes_value = [] } : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [4, 4]} : tensor<16x1xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x1x4x4xf32>) -> tensor<4x4xf32> {
    %0 = const.Declare tensor<2xsi64> = #const.Content<dense<[0, 1]> : tensor<2xsi64>>
    %1 = IE.Squeeze(%arg0, %0) : tensor<1x1x4x4xf32>, tensor<2xsi64> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK: [[VAL0:%.+]] = IE.Squeeze(%arg0) {axes_value = [0, 1]} : tensor<1x1x4x4xf32> -> tensor<4x4xf32>
    // CHECK: return [[VAL0]] : tensor<4x4xf32>
}
