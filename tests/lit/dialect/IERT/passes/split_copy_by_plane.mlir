// RUN: vpux-opt --split-input-file --split-by-planes %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @SplitByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = IERT.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = IERT.SubView %arg0 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = IERT.SubView %arg1 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = IERT.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = IERT.SubView %arg0 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = IERT.SubView %arg1 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = IERT.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[CONCAT:.*]] = IERT.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SplitByHeight(
        %arg0 : memref<1x18x360x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>,
        %arg1 : memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>)
        -> memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}> {
    %0 = IERT.Copy inputs(%arg0 : memref<1x18x360x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>)
                   outputs(%arg1 : memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>)
                   -> memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    return %0 : memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = IERT.SubView %arg0 [0, 0, 0, 0] [1, 18, 255, 1280] :
    // CHECK-SAME:              memref<1x18x360x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}
    // CHECK-SAME:           to memref<1x18x255x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = IERT.SubView %arg1 [0, 0, 0, 0] [1, 18, 255, 1280] :
    // CHECK-SAME:              memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>
    // CHECK-SAME:           to memref<1x18x255x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    // CHECK: %[[COPY_TILE_0:.*]] = IERT.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x18x255x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x18x255x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>)
    // CHECK-SAME:  -> memref<1x18x255x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = IERT.SubView %arg0 [0, 0, 255, 0] [1, 18, 105, 1280] :
    // CHECK-SAME:              memref<1x18x360x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>
    // CHECK-SAME:           to memref<1x18x105x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = IERT.SubView %arg1 [0, 0, 255, 0] [1, 18, 105, 1280] :
    // CHECK-SAME:              memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>
    // CHECK-SAME:           to memref<1x18x105x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    // CHECK: %[[COPY_TILE_1:.*]] = IERT.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x18x105x1280xf16, {order = #NHWC, strides = [14745600, 1, 40960, 32]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x18x105x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>)
    // CHECK-SAME:  -> memref<1x18x105x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    // CHECK: %[[CONCAT:.*]] = IERT.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x18x255x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>,
    // CHECK-SAME:      memref<1x18x105x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>)
    // CHECK-SAME:  -> memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>

    // CHECK: return %[[CONCAT]] : memref<1x18x360x1280xf16, {order = #NHWC, strides = [29491200, 1, 81920, 32]}>
}
