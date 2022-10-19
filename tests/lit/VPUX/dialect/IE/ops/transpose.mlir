// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x16x200x300xf32>) -> tensor<1x16x300x200xf32> {
    %0 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 1, 3, 2]> : tensor<4xsi64>>
    %1 = IE.Transpose(%arg0, %0) :
        tensor<1x16x200x300xf32>, tensor<4xsi64> -> tensor<1x16x300x200xf32>
    return %1 : tensor<1x16x300x200xf32>

    // CHECK:       %[[VAL0:.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<1x16x200x300xf32> -> tensor<1x16x300x200xf32>
    // CHECK:       return %[[VAL0]] : tensor<1x16x300x200xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @FuseTransposes(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x3x16x2xf32> {
    %0 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 1, 3, 2]> : tensor<4xsi64>>
    %1 = IE.Transpose(%arg0, %0) :
        tensor<1x16x2x3xf32>, tensor<4xsi64> -> tensor<1x16x3x2xf32>
    %2 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 2, 1, 3]> : tensor<4xsi64>>
    %3 = IE.Transpose(%1, %2) :
        tensor<1x16x3x2xf32>, tensor<4xsi64> -> tensor<1x3x16x2xf32>
    return %3 : tensor<1x3x16x2xf32>

    // CHECK-NOT: const.Declare
    // CHECK-NOT: IE.Transpose
    // CHECK-NOT: const.Declare
    // CHECK:     [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    // CHECK:     return [[VAL0]] : tensor<1x3x16x2xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d1, d0)>

func @FoldConstTranspose() -> tensor<40x512xf32> {
    %weights = const.Declare tensor<512x40xf32> = #const.Content<dense<1.0> : tensor<512x40xf32>>
    %order = const.Declare tensor<2xsi64> = #const.Content<dense<[1, 0]> : tensor<2xsi64>>
    %1 = IE.Transpose(%weights, %order) : tensor<512x40xf32>, tensor<2xsi64> -> tensor<40x512xf32>

    return %1 : tensor<40x512xf32>

    // CHECK-NOT:  IE.Transpose
    // CHECK:      [[VAL0:%.*]] = const.Declare tensor<40x512xf32>
    // CHECK-SAME:     #const.Content<dense<1.000000e+00> : tensor<512x40xf32>, [#const.Transpose<#map>]>
    // CHECK:      return [[VAL0]] : tensor<40x512xf32>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func @CollapseTransposes(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x28xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 28, 70]} : tensor<1x1x28x70xf16> -> tensor<1x28x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>

    return %2 : tensor<1x70x28xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 70, 28]
    // CHECK-SAME:          } : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func @DoNotCollapseNonTrivialTransposes(%arg0: tensor<1x70x2x28xf16>) -> tensor<1x70x56xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x2x28xf16> -> tensor<1x2x28x70xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 56, 70]} : tensor<1x2x28x70xf16> -> tensor<1x56x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x56x70xf16> -> tensor<1x70x56xf16>

    return %2 : tensor<1x70x56xf16>

    // CHECK:               %[[FIRST_TRANSPOSE:.+]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:              : tensor<1x70x2x28xf16> -> tensor<1x2x28x70xf16>

    // CHECK:               %[[RESHAPE:.+]] = IE.AffineReshape(%[[FIRST_TRANSPOSE]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [0], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 56, 70]
    // CHECK-SAME:          } : tensor<1x2x28x70xf16> -> tensor<1x56x70xf16>

    // CHECK:               %[[LAST_TRANSPOSE:.+]] = IE.Transpose(%[[RESHAPE]]) {order_value = #map}
    // CHECK-SAME:              : tensor<1x56x70xf16> -> tensor<1x70x56xf16>

    // CHECK:       return %[[LAST_TRANSPOSE]] : tensor<1x70x56xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func @CollapseTransposesWithSqueeze(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x28xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.Squeeze(%0) { axes_value = [0] } : tensor<1x1x28x70xf16> -> tensor<1x28x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>

    return %2 : tensor<1x70x28xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 70, 28]
    // CHECK-SAME:          } : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func @CollapseTransposesWithReshape(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x28xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.Reshape(%0) { shape_value = [1, 28, 70] } : tensor<1x1x28x70xf16> -> tensor<1x28x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>

    return %2 : tensor<1x70x28xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 70, 28]
    // CHECK-SAME:          } : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x70x28xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func @CollapseTransposesWithUnsqueeze(%arg0: tensor<1x28x70xf16>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>
    %1 = IE.Unsqueeze(%0) {axes_value = [0]} : tensor<1x70x28xf16> -> tensor<1x1x70x28xf16>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x1x70x28xf16> -> tensor<1x1x28x70xf16>

    return %2 : tensor<1x1x28x70xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0, 1], [2], [3]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 1, 28, 70]
    // CHECK-SAME:          } : tensor<1x28x70xf16> -> tensor<1x1x28x70xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x1x28x70xf16>
}
