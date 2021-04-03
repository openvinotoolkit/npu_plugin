// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --add-layouts-and-strides %s | FileCheck %s

//
// The 'add-layouts-and-strides' pass:
//
//   * Adds default layout and strides to the memref
//   TODO: Update and enable the pass to propagate user layout. [Track number: #6185]
//

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 4 + d2)>

// CHECK: func @SingleLayer([[ARG0:%arg[0-9]*]]: memref<1x8x4xf16, #map>, [[ARG1:%arg[0-9]*]]: memref<1x8x4xf16, #map>) {
func @SingleLayer(%arg0: memref<1x8x4xf16>, %arg1: memref<1x8x4xf16>) {
    IERT.SoftMax(%arg0, %arg1) {axisInd = 1 : i32} : memref<1x8x4xf16>, memref<1x8x4xf16>
    return

    // CHECK: IERT.SoftMax([[ARG0]], [[ARG1]]) {axisInd = 1 : i32} : memref<1x8x4xf16, #map>, memref<1x8x4xf16, #map>
    // CHECK: return
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0 * 1000 + d1)>

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16, #map>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16, #map>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = memref.alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16, "DDR">

    %1 = memref.alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">
    memref.dealloc %0 : memref<1x1000xf16, "DDR">

    %2 = memref.alloc() : memref<1x1000xf16, "DDR">
    IERT.SoftMax(%1, %2) {axisInd = 1 : i32} : memref<1x1000xf16, "DDR">, memref<1x1000xf16, "DDR">
    memref.dealloc %1 : memref<1x1000xf16, "DDR">

    linalg.copy(%2, %arg1) : memref<1x1000xf16, "DDR">, memref<1x1000xf16>
    memref.dealloc %2 : memref<1x1000xf16, "DDR">

    return

    // CHECK: %0 = memref.alloc() : memref<1x1000xf16, #map, "DDR">
    // CHECK: IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16, #map>, memref<1x1000xf16, #map, "DDR">

    // CHECK: %1 = memref.alloc() : memref<1x1000xf16, #map, "DDR">
    // CHECK: IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16, #map, "DDR">, memref<1x1000xf16, #map, "DDR">
    // CHECK: memref.dealloc %0 : memref<1x1000xf16, #map, "DDR">

    // CHECK: %2 = memref.alloc() : memref<1x1000xf16, #map, "DDR">
    // CHECK: IERT.SoftMax(%1, %2) {axisInd = 1 : i32} : memref<1x1000xf16, #map, "DDR">, memref<1x1000xf16, #map, "DDR">
    // CHECK: memref.dealloc %1 : memref<1x1000xf16, #map, "DDR">

    // CHECK: linalg.copy(%2, %arg1) : memref<1x1000xf16, #map, "DDR">, memref<1x1000xf16, #map>
    // CHECK: memref.dealloc %2 : memref<1x1000xf16, #map, "DDR">
    // CHECK: return
}
