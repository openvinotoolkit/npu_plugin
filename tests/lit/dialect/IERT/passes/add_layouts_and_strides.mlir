// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --add-layouts-and-strides %s | FileCheck %s

//
// The 'add-layouts-and-strides' pass:
//
//   * Adds default layout and strides to the memref
//   TODO: Update and enable the pass to propagate user layout. [Track number: #6185]
//

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 4 + d2)>

// CHECK: func @SingleLayer([[ARG0:%arg[0-9]*]]: memref<1x8x4xf16, #map>, [[ARG1:%arg[0-9]*]]: memref<1x8x4xf16, #map>) -> memref<1x8x4xf16, #map> {
func @SingleLayer(%arg0: memref<1x8x4xf16>, %arg1: memref<1x8x4xf16>) -> memref<1x8x4xf16> {
    %0 = IERT.SoftMax {axisInd = 2 : i32} inputs(%arg0 : memref<1x8x4xf16>) outputs(%arg1 : memref<1x8x4xf16>) -> memref<1x8x4xf16>
    return %0 : memref<1x8x4xf16>

    // CHECK: [[VAR0:%.*]] = IERT.SoftMax {axisInd = 2 : i32} inputs(%arg0 : memref<1x8x4xf16, #map>) outputs(%arg1 : memref<1x8x4xf16, #map>) -> memref<1x8x4xf16, #map>
    // CHECK: return [[VAR0]] : memref<1x8x4xf16, #map>
}
