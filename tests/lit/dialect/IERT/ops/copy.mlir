// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldCopy
func @FoldCopy(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = memref.alloc() : memref<1x8x4x2xf16>
    %1 = IERT.Copy inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%0 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %2 = memref.alloc() : memref<1x8x4x2xf16>
    %3 = IERT.SoftMax {axisInd = 1} inputs(%1 : memref<1x8x4x2xf16>) outputs(%2 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %4 = memref.alloc() : memref<1x8x4x2xf16>
    %5 = IERT.Copy inputs(%3 : memref<1x8x4x2xf16>) outputs(%4 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %6 = memref.alloc() : memref<1x8x4x2xf16>
    %7 = IERT.SoftMax {axisInd = 1} inputs(%5 : memref<1x8x4x2xf16>) outputs(%6 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %8 = IERT.Copy inputs(%7 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %8 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR3:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs([[VAR2]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: return [[VAR4]] : memref<1x8x4x2xf16>
}

// -----

func @FuseCopies(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = memref.alloc() : memref<1x8x4x2xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%0 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %2 = memref.alloc() : memref<1x8x4x2xf16>
    %3 = IERT.Copy inputs(%1 : memref<1x8x4x2xf16>) outputs(%2 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %4 = IERT.Copy inputs(%3 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %4 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: return [[VAR2]] : memref<1x8x4x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 9 + d1 * 3 + d2 + d3)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 9 + d1 * 3 + d2 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 9 + d1 * 3 + d2 + d3 + 432)>

// CHECK-LABEL: @FoldSubViewCopy
func @FoldSubViewCopy(%arg0: memref<16x1x3x3xf16, #NHWC, #map0>) -> memref<16x1x3x3xf16, #NHWC, #map0> {
    %0 = const.Declare memref<64x1x3x3xf16, #NHWC, #map0> =
        #const.Content<dense<1.000000e+00> : tensor<64x1x1x3x3xf16>,
        [#const.Reshape<[64, 1, 3, 3]>, #const.Reorder<#NHWC>]>
    %1 = IERT.SubView %0 [48, 0, 0, 0] [16, 1, 3, 3] : memref<64x1x3x3xf16, #NHWC, #map0> -> memref<16x1x3x3xf16, #NHWC, #map1>
    %2 = memref.alloc() : memref<16x1x3x3xf16, #NHWC, #map0>
    %3 = IERT.Copy
        inputs(%1 : memref<16x1x3x3xf16, #NHWC, #map1>)
        outputs(%2 : memref<16x1x3x3xf16, #NHWC, #map0>)
        -> memref<16x1x3x3xf16, #NHWC, #map0>
    %4 = IERT.ReLU
        inputs(%3 : memref<16x1x3x3xf16, #NHWC, #map0>)
        outputs(%arg0 : memref<16x1x3x3xf16, #NHWC, #map0>)
        -> memref<16x1x3x3xf16, #NHWC, #map0>
    return %4 : memref<16x1x3x3xf16, #NHWC, #map0>

    // CHECK:       [[CONST_DECLARE:%.*]] = const.Declare memref<16x1x3x3xf16, #NHWC, #map> =
    // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<64x1x1x3x3xf16>
    // CHECK-SAME:      #const.Reshape<[64, 1, 3, 3]>
    // CHECK-SAME:      #const.Reorder<#NHWC>
    // CHECK-SAME:      #const.SubView<[48, 0, 0, 0], [16, 1, 3, 3]>

    // CHECK:       [[RELU:%.*]] = IERT.ReLU
    // CHECK-SAME:      inputs([[CONST_DECLARE]] : memref<16x1x3x3xf16, #NHWC, #map>)
    // CHECK-SAME:      outputs(%arg0 : memref<16x1x3x3xf16, #NHWC, #map>)
    // CHECK-SAME:      -> memref<16x1x3x3xf16, #NHWC, #map>

    // CHECK:       return [[RELU]] : memref<16x1x3x3xf16, #NHWC, #map>
}
