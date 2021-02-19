// RUN: vpux-opt --split-input-file --convert-tile-to-per-axis-tiles %s | FileCheck %s

// CHECK-LABEL: @TileOpByMultilyAxes2PerAxisTiles
func @TileOpByMultilyAxes2PerAxisTiles(%arg0: tensor<3x4x2xf32>) -> tensor<9x4x4xf32> {
    %0 = IE.Constant tensor<3xsi64> = dense<[3, 1, 2]> : tensor<3xsi64>
    %1 = IE.Tile(%arg0, %0) : tensor<3x4x2xf32>, tensor<3xsi64> -> tensor<9x4x4xf32>
    // CHECK-NOT:   IE.Tile
    // CHECK:       %[[VAL0:.*]] = IE.PerAxisTile(%arg0) {axis = 0 : i32, tiles = 3 : i32} : tensor<3x4x2xf32> -> tensor<9x4x2xf32>
    // CHECK:       %[[VAL1:.*]] = IE.PerAxisTile(%[[VAL0:.*]]) {axis = 2 : i32, tiles = 2 : i32} : tensor<9x4x2xf32> -> tensor<9x4x4xf32>

    return %1 : tensor<9x4x4xf32>
    // CHECK:       return %[[VAL1]]
}
