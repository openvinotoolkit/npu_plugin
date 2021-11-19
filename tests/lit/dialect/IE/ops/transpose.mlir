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
