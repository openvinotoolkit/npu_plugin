// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldSubTensor
func @FoldSubTensor() -> tensor<2x2xf32> {
    %0 = const.Declare tensor<4x4xf32> = #const.Content<
        dense<1.000000e+00> : tensor<4x4xf32>
    >
    %1 = tensor.extract_slice %0[1, 1] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
    return %1 : tensor<2x2xf32>

    // CHECK:       [[VAR0:%.+]] = const.Declare tensor<2x2xf32> = #const.Content<
    // CHECK-SAME:      dense<1.000000e+00> : tensor<4x4xf32>
    // CHECK-SAME:      [#const.SubView<[1, 1], [2, 2]>]

    // CHECK-NOT:   tensor.extract_slice

    // CHECK:       return [[VAR0]] : tensor<2x2xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0 * 4 + d1 + 5)>

// CHECK-LABEL: @FoldSubView
func @FoldSubView() -> memref<2x2xf32> {
    %0 = const.Declare memref<4x4xf32> = #const.Content<
        dense<1.000000e+00> : tensor<4x4xf32>
    >
    %1 = memref.subview %0[1, 1] [2, 2] [1, 1] : memref<4x4xf32> to memref<2x2xf32, #map>
    %2 = memref.clone %1 : memref<2x2xf32, #map> to memref<2x2xf32>
    return %2 : memref<2x2xf32>

    // CHECK:       [[VAR0:%.+]] = const.Declare memref<2x2xf32> = #const.Content<
    // CHECK-SAME:      dense<1.000000e+00> : tensor<4x4xf32>
    // CHECK-SAME:      [#const.SubView<[1, 1], [2, 2]>]

    // CHECK-NOT:   memref.subview

    // CHECK:       [[VAR1:%.+]] = memref.clone [[VAR0]] : memref<2x2xf32> to memref<2x2xf32>

    // CHECK:       return [[VAR1]] : memref<2x2xf32>
}
