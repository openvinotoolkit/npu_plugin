// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

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

func @FoldTranspose(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x16x2x3xf32> {
    %0 = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 1, 2, 3]> : tensor<4xsi64>>
    %1 = IE.Transpose(%arg0, %0) :
        tensor<1x16x2x3xf32>, tensor<4xsi64> -> tensor<1x16x2x3xf32>
    return %1 : tensor<1x16x2x3xf32>

    // CHECK-NOT: IE.Constant
    // CHECK-NOT: IE.Transpose
    // CHECK:     return %arg0 : tensor<1x16x2x3xf32>
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
