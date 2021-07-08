// RUN: vpux-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldTile
func @FoldTile(%arg0: tensor<3x4x2xf32>) -> tensor<3x4x2xf32> {
    %0 = const.Declare tensor<3xsi64> = #const.Content<dense<1> : tensor<3xsi64>>
    %1 = IE.Tile(%arg0, %0) : tensor<3x4x2xf32>, tensor<3xsi64> -> tensor<3x4x2xf32>
    // CHECK-NOT:   IE.Tile
    return %1 : tensor<3x4x2xf32>
    // CHECK:       return %arg0
}

// CHECK-LABEL: @InsertUnsqueezeBeforedTile
func @InsertUnsqueezeBeforedTile(%arg0: tensor<2x3xf32>) -> tensor<1x6x15xf32> {
    %0 = const.Declare tensor<3xsi64> = #const.Content<dense<[1, 3, 5]> : tensor<3xsi64>>
    // CHECK:       %[[VAL0:.*]] = const.Declare tensor<3xsi64> = #const.Content<dense<[1, 3, 5]> : tensor<3xsi64>>
    // CHECK:       %[[VAL1:.*]] = IE.Unsqueeze(%arg0) {axes_value = [0]} : tensor<2x3xf32> -> tensor<1x2x3xf32>
    %1 = IE.Tile(%arg0, %0) : tensor<2x3xf32>, tensor<3xsi64> -> tensor<1x6x15xf32>
    // CHECK:       %[[VAL2:.*]] = IE.Tile(%[[VAL1]], %[[VAL0]]) : tensor<1x2x3xf32>, tensor<3xsi64> -> tensor<1x6x15xf32>

    return %1 : tensor<1x6x15xf32>
    // CHECK:       return %[[VAL2]]
}
