// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --convert-depthToSpace %s | FileCheck %s

// CHECK-LABEL: @Depth2SpaceCanConvertToNNDMAs_BLOCKS_FIRST
func @Depth2SpaceCanConvertToNNDMAs_BLOCKS_FIRST(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x256x256xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "BLOCKS_FIRST"} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    // CHECK:       [[DepthToSpace:%.*]] = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "BLOCKS_FIRST"} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>
    // CHECK:       return [[DepthToSpace]] : tensor<1x1x256x256xf16>
}

// CHECK-LABEL: @Depth2SpaceCannotConvertToNNDMAsWithLargeHeight_BLOCKS_FIRST
func @Depth2SpaceCannotConvertToNNDMAsWithLargeHeight_BLOCKS_FIRST(%arg0: tensor<1x4x512x8xf16>) -> tensor<1x1x1024x16xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x4x512x8xf16> -> tensor<1x1x1024x16xf16>

    return %0 : tensor<1x1x1024x16xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 2, 2, 1, 512, 8]} : tensor<1x4x512x8xf16> -> tensor<1x2x2x1x512x8xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #map0} : tensor<1x2x2x1x512x8xf16> -> tensor<1x1x512x2x8x2xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [1, 1, 1024, 16]} : tensor<1x1x512x2x8x2xf16> -> tensor<1x1x1024x16xf16>
    // CHECK:       return [[VAL2]] : tensor<1x1x1024x16xf16>
}

// CHECK-LABEL: @Depth2SpaceCanConvertToNNDMAs_DEPTH_FIRST
func @Depth2SpaceCanConvertToNNDMAs_DEPTH_FIRST(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x256x256xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    // CHECK:       [[DepthToSpace:%.*]] = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x16x64x64xf16> -> tensor<1x1x256x256xf16>
    // CHECK:       return [[DepthToSpace]] : tensor<1x1x256x256xf16>
}

// CHECK-LABEL: @Depth2SpaceCannotConvertToNNDMAsWithLargeSize_DEPTH_FIRST
func @Depth2SpaceCannotConvertToNNDMAsWithLargeSize_DEPTH_FIRST(%arg0: tensor<1x64x256x256xf16>) -> tensor<1x4x1024x1024xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x64x256x256xf16> -> tensor<1x4x1024x1024xf16>

    return %0 : tensor<1x4x1024x1024xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 4, 4, 4, 256, 256]} : tensor<1x64x256x256xf16> -> tensor<1x4x4x4x256x256xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #map1} : tensor<1x4x4x4x256x256xf16> -> tensor<1x4x256x4x256x4xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [1, 4, 1024, 1024]} : tensor<1x4x256x4x256x4xf16> -> tensor<1x4x1024x1024xf16>
    // CHECK:       return [[VAL2]] : tensor<1x4x1024x1024xf16>
}