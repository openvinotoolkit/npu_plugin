//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-tile-to-per-axis-tiles %s | FileCheck %s

// CHECK-LABEL: @TileOpByMultilyAxes2PerAxisTiles
func @TileOpByMultilyAxes2PerAxisTiles(%arg0: tensor<3x4x2xf32>) -> tensor<9x4x4xf32> {
    %0 = const.Declare tensor<3xsi64> = dense<[3, 1, 2]> : tensor<3xsi64>
    %1 = IE.Tile(%arg0, %0) : tensor<3x4x2xf32>, tensor<3xsi64> -> tensor<9x4x4xf32>
    // CHECK-NOT:   IE.Tile
    // CHECK:       %[[VAL0:.*]] = IE.PerAxisTile(%arg0)
    // CHECK-SAME:      axis = 0
    // CHECK-SAME:      tiles = 3
    // CHECK:       %[[VAL1:.*]] = IE.PerAxisTile(%[[VAL0:.*]])
    // CHECK-SAME:      axis = 2
    // CHECK-SAME:      tiles = 2

    return %1 : tensor<9x4x4xf32>
    // CHECK:       return %[[VAL1]]
}
