// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-to-mem-permute --canonicalize %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d3, d5, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d2, d3, d1, d4)>

// CHECK-LABEL: @ConvertToMemPermuteNotCanon
func @ConvertToMemPermuteNotCanon(%arg0: tensor<1x360x640x20x2x2xf16, {order = #map0}>) -> (tensor<1x2x640x20x360x2xf16>) {
    %0 = IE.Reorder(%arg0) {dstOrder = #map3} : tensor<1x360x640x20x2x2xf16, {order = #map0}> -> tensor<1x360x640x20x2x2xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #map1, mem_perm = #map3} : tensor<1x360x640x20x2x2xf16> -> tensor<1x2x640x20x360x2xf16, {order = #map1}>
    %2 = IE.Reorder(%1) {dstOrder = #map3} : tensor<1x2x640x20x360x2xf16, {order = #map1}> -> tensor<1x2x640x20x360x2xf16>
    
    return %2 : tensor<1x2x640x20x360x2xf16>

    // CHECK-NOT: IE.Reorder
    // CHECK:     %[[VAL0:.*]] = IE.MemPermute(%arg0) {dst_order = #map1, mem_perm = #map2} : tensor<1x360x640x20x2x2xf16, {order = #map0}> -> tensor<1x2x640x20x360x2xf16, {order = #map1}>
    // CHECK:     %[[VAL1:.*]] = IE.MemPermute(%[[VAL0]]) {dst_order = #map3, mem_perm = #map4} : tensor<1x2x640x20x360x2xf16, {order = #map1}> -> tensor<1x2x640x20x360x2xf16>
    // CHECK:     return %[[VAL1]] : tensor<1x2x640x20x360x2xf16>
}
