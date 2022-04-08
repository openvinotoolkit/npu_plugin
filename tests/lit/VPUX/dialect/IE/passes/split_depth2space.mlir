// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --split-depth-to-space %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SplitDepth2SpaceWithDepthFirst
func @SplitDepth2SpaceWithDepthFirst(%arg0: tensor<1x80x38x320xf16, {order = #NHWC}>) -> tensor<1x20x76x640xf16, {order = #NHWC}> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = "DEPTH_FIRST"} : tensor<1x80x38x320xf16, {order = #NHWC}> -> tensor<1x20x76x640xf16, {order = #NHWC}>

    return %0 : tensor<1x20x76x640xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[SLICE_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_0:%.*]] = IE.DepthToSpace([[SLICE_0]]) {block_size = 2 : i64, mode = "DEPTH_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_1:%.*]] = IE.Slice %arg0 [0, 0, 8, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_1:%.*]] = IE.DepthToSpace([[SLICE_1]]) {block_size = 2 : i64, mode = "DEPTH_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_2:%.*]] = IE.Slice %arg0 [0, 0, 16, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_2:%.*]] = IE.DepthToSpace([[SLICE_2]]) {block_size = 2 : i64, mode = "DEPTH_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 24, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_3:%.*]] = IE.DepthToSpace([[SLICE_3]]) {block_size = 2 : i64, mode = "DEPTH_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 32, 0] [1, 80, 6, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x6x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_4:%.*]] = IE.DepthToSpace([[SLICE_4]]) {block_size = 2 : i64, mode = "DEPTH_FIRST"} : tensor<1x80x6x320xf16, {order = #NHWC}> -> tensor<1x20x12x640xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[DEPTHTOSPACE_0]], [[DEPTHTOSPACE_1]], [[DEPTHTOSPACE_2]], [[DEPTHTOSPACE_3]], [[DEPTHTOSPACE_4]])
    // CHECK:                               {per_axis = {axis = 2 : i64}} : tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x12x640xf16, {order = #NHWC}> -> tensor<1x20x76x640xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x20x76x640xf16, {order = #NHWC}>
}

// CHECK-LABEL: @SplitDepth2SpaceWithBlockFirst
func @SplitDepth2SpaceWithBlockFirst(%arg0: tensor<1x80x38x320xf16, {order = #NHWC}>) -> tensor<1x20x76x640xf16, {order = #NHWC}> {
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x80x38x320xf16, {order = #NHWC}> -> tensor<1x20x76x640xf16, {order = #NHWC}>

    return %0 : tensor<1x20x76x640xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.DepthToSpace
    // CHECK:       [[SLICE_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_0:%.*]] = IE.DepthToSpace([[SLICE_0]]) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_1:%.*]] = IE.Slice %arg0 [0, 0, 8, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_1:%.*]] = IE.DepthToSpace([[SLICE_1]]) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_2:%.*]] = IE.Slice %arg0 [0, 0, 16, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_2:%.*]] = IE.DepthToSpace([[SLICE_2]]) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_3:%.*]] = IE.Slice %arg0 [0, 0, 24, 0] [1, 80, 8, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x8x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_3:%.*]] = IE.DepthToSpace([[SLICE_3]]) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x80x8x320xf16, {order = #NHWC}> -> tensor<1x20x16x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_4:%.*]] = IE.Slice %arg0 [0, 0, 32, 0] [1, 80, 6, 320] : tensor<1x80x38x320xf16, {order = #NHWC}> to tensor<1x80x6x320xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHTOSPACE_4:%.*]] = IE.DepthToSpace([[SLICE_4]]) {block_size = 2 : i64, mode = "BLOCKS_FIRST"} : tensor<1x80x6x320xf16, {order = #NHWC}> -> tensor<1x20x12x640xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[DEPTHTOSPACE_0]], [[DEPTHTOSPACE_1]], [[DEPTHTOSPACE_2]], [[DEPTHTOSPACE_3]], [[DEPTHTOSPACE_4]])
    // CHECK:                               {per_axis = {axis = 2 : i64}} : tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x16x640xf16, {order = #NHWC}>, tensor<1x20x12x640xf16, {order = #NHWC}> -> tensor<1x20x76x640xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x20x76x640xf16, {order = #NHWC}>
}
