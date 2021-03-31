// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x16x200x300xf32>) -> tensor<1x16x300x200xf32> {
    %0 = IE.Constant tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>
    %1 = IE.Transpose(%arg0, %0) :
        tensor<1x16x200x300xf32>, tensor<4xsi64> -> tensor<1x16x300x200xf32>
    return %1 : tensor<1x16x300x200xf32>

    // CHECK:       %[[VAL0:.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<1x16x200x300xf32> -> tensor<1x16x300x200xf32>
    // CHECK:       return %[[VAL0]] : tensor<1x16x300x200xf32>
}
