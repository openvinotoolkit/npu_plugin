// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 12 + d2 * 3 + d3)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 48 + d2 * 3 + d3)>

// CHECK-LABEL: @Fold
func @Fold(%arg0: memref<1x3x8x4xf32, #NHWC, #map>) -> memref<1x3x8x4xf32, #NHWC, #map> {
    %0 = const.Declare memref<1x3x16x16xf32, #NHWC, #map0> =
        #const.Content<dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]>

    %1 = IERT.SubView %0 [0, 0, 8, 12] [1, 3, 8, 4] :
        memref<1x3x16x16xf32, #NHWC, #map0> to
        memref<1x3x8x4xf32, #NHWC, #map0>

    %2 = IERT.Copy
        inputs(%1 : memref<1x3x8x4xf32, #NHWC, #map0>)
        outputs(%arg0 : memref<1x3x8x4xf32, #NHWC, #map>)
        -> memref<1x3x8x4xf32, #NHWC, #map>

    return %2 : memref<1x3x8x4xf32, #NHWC, #map>

    // CHECK:       [[VAR0:%.+]] = const.Declare memref<1x3x8x4xf32, #NHWC, #map> =
    // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 8, 12], [1, 3, 8, 4]>]
    // CHECK-NOT:   IERT.SubView

    // CHECK:       [[VAR1:%.+]] = IERT.Copy
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x3x8x4xf32, #NHWC, #map>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x3x8x4xf32, #NHWC, #map>)

    // CHECK:       return [[VAR1]] : memref<1x3x8x4xf32, #NHWC, #map>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 256 + d2 * 16 + d3)>

// CHECK-LABEL: @ComposeSubView
func @ComposeSubView(%arg0: memref<1x3x8x4xf32>) -> memref<1x3x8x4xf32> {
    %0 = memref.alloc() : memref<1x3x16x16xf32>

    %1 = IERT.SubView %0 [0, 0, 0, 8] [1, 3, 16, 8] :
        memref<1x3x16x16xf32> to
        memref<1x3x16x8xf32, #map>

    %2 = IERT.SubView %1 [0, 0, 8, 4] [1, 3, 8, 4] :
        memref<1x3x16x8xf32, #map> to
        memref<1x3x8x4xf32, #map>

    %3 = IERT.ReLU
        inputs(%2 : memref<1x3x8x4xf32, #map>)
        outputs(%arg0 : memref<1x3x8x4xf32>)
        -> memref<1x3x8x4xf32>

    return %3 : memref<1x3x8x4xf32>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x3x16x16xf32>

    // CHECK:       [[VAR1:%.*]] = IERT.SubView [[VAR0]] [0, 0, 8, 12] [1, 3, 8, 4] :
    // CHECK-SAME:      memref<1x3x16x16xf32> to memref<1x3x8x4xf32, #map>

    // CHECK:       [[VAR2:%.*]] = IERT.ReLU
    // CHECK-SAME:      inputs([[VAR1]] : memref<1x3x8x4xf32, #map>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x3x8x4xf32>)
    // CHECK-SAME:      -> memref<1x3x8x4xf32>

    // CHECK:       return [[VAR2]] : memref<1x3x8x4xf32>
}
