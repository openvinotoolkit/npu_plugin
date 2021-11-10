// RUN: vpux-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @Eliminate
func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [4, 4] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: IE.AffineReshape
    // CHECK:     return %arg0
}

// CHECK-LABEL: @ConstFold
func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<16xf32> = #const.Content<dense<1.0> : tensor<16xf32>>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0, 1]], shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK:       [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<16xf32>, [#const.Reshape<[4, 4]>]>
    // CHECK-NOT:   IE.AffineReshape
    // CHECK:       return [[VAL0]]
}

// CHECK-LABEL: @FuseWithReshape
func @FuseWithReshape(%arg0: tensor<15x2xf32>) -> tensor<10x3x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [30, 1] } : tensor<15x2xf32> -> tensor<30x1xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0, 1], [2]], shape_value = [10, 3, 1] } : tensor<30x1xf32> -> tensor<10x3x1xf32>

    return %1 : tensor<10x3x1xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [10, 3, 1]} : tensor<15x2xf32> -> tensor<10x3x1xf32>
    // CHECK: return [[VAL0]] : tensor<10x3x1xf32>
}

// CHECK-LABEL: @FuseWithSqueeze
func @FuseWithSqueeze(%arg0: tensor<15x2x1xf32>) -> tensor<30xf32> {
    %0 = IE.Squeeze(%arg0) { axes_value = [2] } : tensor<15x2x1xf32> -> tensor<15x2xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf32> -> tensor<30xf32>

    return %1 : tensor<30xf32>

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0]], shape_value = [30]} : tensor<15x2x1xf32> -> tensor<30xf32>
    // CHECK: return [[VAL0]] : tensor<30xf32>
}

// CHECK-LABEL: @FuseWithUnsqueeze
func @FuseWithUnsqueeze(%arg0: tensor<15x2xf32>) -> tensor<1x30xf32> {
    %0 = IE.Unsqueeze(%arg0) { axes_value = [0, 1] } : tensor<15x2xf32> -> tensor<1x1x15x2xf32>
    %1 = IE.AffineReshape(%0) { dim_mapping = [[0], [0], [1], [1]], shape_value = [1, 30] } : tensor<1x1x15x2xf32> -> tensor<1x30xf32>

    return %1 : tensor<1x30xf32>

    // CHECK: [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 30]} : tensor<15x2xf32> -> tensor<1x30xf32>
    // CHECK: return [[VAL0]] : tensor<1x30xf32>
}
