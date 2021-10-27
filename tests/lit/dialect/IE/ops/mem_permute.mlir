// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @FoldMemPermute
func @FoldMemPermute(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x16x2x3xf32> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x16x2x3xf32> -> tensor<1x16x2x3xf32>
    return %0 : tensor<1x16x2x3xf32>

    // CHECK-NOT: IE.MemPermute
    // CHECK:     return %arg0 : tensor<1x16x2x3xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @FuseMemPermutes
func @FuseMemPermutes(%arg0: tensor<1x16x2x3xf32>, %arg1: tensor<1x16x2x3xf32, {order = #NHWC}>) ->
        (tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map1} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #map2} :
        tensor<1x3x16x2xf32> -> tensor<1x3x2x16xf32>

    %2 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #map1} :
        tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #map1} :
        tensor<1x3x16x2xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32>

    return %1, %3 : tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>

    // CHECK-NOT: IE.MemPermute
    // CHECK-NOT: IE.MemPermute
    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} : tensor<1x16x2x3xf32> -> tensor<1x3x2x16xf32>
    // CHECK:     %[[VAL_1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32>
    // CHECK:     return %[[VAL_0]], %[[VAL_1]] : tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @ConvertToPermuteCast
func @ConvertToPermuteCast(%arg0: tensor<1x100x1x1xf32>, %arg1: tensor<1x100x1x1xf32, {order = #NHWC}>) ->
        (tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} :
        tensor<1x100x1x1xf32> -> tensor<1x1x100x1xf32>

    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} :
            tensor<1x100x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x100xf32>

    return %0, %1 : tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>

    //CHECK:     %[[VAL_0:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #map} : tensor<1x100x1x1xf32> -> tensor<1x1x100x1xf32>
    //CHECK:     %[[VAL_1:.*]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x100x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x100xf32>
    //CHECK:     return %[[VAL_0]], %[[VAL_1]] : tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL:   @FusePermCastAndMemPerm
func @FusePermCastAndMemPerm(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) ->
            tensor<1x1x1000x1xf32> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #map} :
            tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #map} :
        tensor<1x1x1000x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>

    return %1 : tensor<1x1x1000x1xf32>

    // CHECK:     %[[VAL_0:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    // CHECK:     return %[[VAL_0]] : tensor<1x1x1000x1xf32>
}
