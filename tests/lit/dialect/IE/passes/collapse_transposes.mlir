// RUN: vpux-opt --split-input-file --collapse-transposes-pass --canonicalize %s | FileCheck %s

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
