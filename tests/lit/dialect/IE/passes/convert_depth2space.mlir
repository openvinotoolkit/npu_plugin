// RUN: vpux-opt --split-input-file --convert-depthToSpace %s | FileCheck %s

// CHECK-LABEL: @ConvertDepth2SpaceLayer_BLOCKS_FIRST
func @ConvertDepth2SpaceLayer_BLOCKS_FIRST(%arg0: tensor<1x16x320x180xf16>) -> tensor<1x1x1280x720xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "BLOCKS_FIRST"} : tensor<1x16x320x180xf16> -> tensor<1x1x1280x720xf16>
    
    return %0 : tensor<1x1x1280x720xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 4, 4, 1, 320, 180]} : tensor<1x16x320x180xf16> -> tensor<1x4x4x1x320x180xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #map0} : tensor<1x4x4x1x320x180xf16> -> tensor<1x1x320x4x180x4xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [1, 1, 1280, 720]} : tensor<1x1x320x4x180x4xf16> -> tensor<1x1x1280x720xf16>
    // CHECK:       return [[VAL2]]
}

// CHECK-LABEL: func @ConvertDepth2SpaceLayer_DEPTH_FIRST
func @ConvertDepth2SpaceLayer_DEPTH_FIRST(%arg0: tensor<1x16x320x180xf16>) -> tensor<1x1x1280x720xf16> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = "DEPTH_FIRST"} : tensor<1x16x320x180xf16> -> tensor<1x1x1280x720xf16>
    
    return %0 : tensor<1x1x1280x720xf16>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[VAL0:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 4, 4, 320, 180]} : tensor<1x16x320x180xf16> -> tensor<1x1x4x4x320x180xf16>
    // CHECK:       [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #map1} : tensor<1x1x4x4x320x180xf16> -> tensor<1x1x320x4x180x4xf16>
    // CHECK:       [[VAL2:%.*]] = IE.Reshape([[VAL1]]) {shape_value = [1, 1, 1280, 720]} : tensor<1x1x320x4x180x4xf16> -> tensor<1x1x1280x720xf16>
    // CHECK:       return [[VAL2]]
}
