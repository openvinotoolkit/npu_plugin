//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --tile-act-shave-kernel-task --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileStridedMVN(%arg0: memref<1x128x64x32xf16, #NWHC>)
        -> memref<1x128x64x32xf16, #NWHC> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>) outputs(%0 : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_MVN inputs(%1 as %arg1: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    return %4: memref<1x128x64x32xf16, #NWHC>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>

    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   [[MVN:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_MVN inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16, #NWHC>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileStridedMVNWithDifferentTileSize(%arg0: memref<1x129x64x32xf16, #NWHC>)
        -> memref<1x129x64x32xf16, #NWHC> {
    %0 = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x129x64x32xf16, #NWHC>) outputs(%0 : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_MVN inputs(%1 as %arg1: memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>, memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x129x64x32xf16, #NWHC>
    %4 = VPUIP.Copy inputs(%results : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%3 : memref<1x129x64x32xf16, #NWHC>) -> memref<1x129x64x32xf16, #NWHC>
    return %4: memref<1x129x64x32xf16, #NWHC>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x129x64x32xf16, #NWHC>) outputs([[INPUT_CMX]] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_CMX:%.*]] = memref.alloc() : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 65, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 65, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 65, 0, 0] [1, 64, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 65, 0, 0] [1, 64, 64, 32] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   [[MVN:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_MVN inputs([[SUBVIEW0]] as %arg1: memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>){
    // CHECK:                       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:                       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : memref<1x65x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [264192, 1, 129, 8256]}, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX]] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x129x64x32xf16, #NWHC>
    // CHECK:   [[COPY_BACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x129x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x129x64x32xf16, #NWHC>) -> memref<1x129x64x32xf16, #NWHC>
    // CHECK:   return [[COPY_BACK]] : memref<1x129x64x32xf16, #NWHC>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, HALF_PIXEL = 0
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[INBUF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[COPY1]] : memref<1x1x192x320xf16>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileHalfPixelInterpolateNotOnOuterMostDim(%arg0: memref<1x3x96x160xf16, [@CMX_NN, 0]>) -> memref<1x3x192x320xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, HALF_PIXEL = 0
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg2: memref<1x3x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x3x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x3x96x160xf16, [@CMX_NN, 0]>, memref<1x3x192x320xf16, [@CMX_NN, 0]>
    }
    return %results : memref<1x3x192x320xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 49, 160] : memref<1x3x96x160xf16, [@CMX_NN, 0]> to memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF_0:%.*]] = memref.alloc() : memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[INPUT_BUF_0]] : memref<1x3x49x160xf16, [@CMX_NN, 0]>) -> memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFF_0:%.*]] = memref.alloc() : memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 3, 49, 160] : memref<1x3x96x160xf16, [@CMX_NN, 0]> to memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INPUT_BUF_1:%.*]] = memref.alloc() : memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x3x49x160xf16, {order = #NCHW, strides = [46080, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[INPUT_BUF_1]] : memref<1x3x49x160xf16, [@CMX_NN, 0]>) -> memref<1x3x49x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUFF_1:%.*]] = memref.alloc() : memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Interpolate inputs([[COPY_0]] as %arg1: memref<1x3x49x160xf16, [@CMX_NN, 0]>, [[COPY_1]] as %arg2: memref<1x3x49x160xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFF_0]] as %arg3: memref<1x3x96x320xf16, [@CMX_NN, 0]>, [[OUTPUT_BUFF_1]] as %arg4: memref<1x3x96x320xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x96x320xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x3x49x160xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x3x49x160xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUFF:%.*]] = memref.alloc() : memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 3, 96, 320] : memref<1x3x192x320xf16, [@CMX_NN, 0]> to memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy inputs([[INTERP]]#0 : memref<1x3x96x320xf16, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) -> memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 96, 0] [1, 3, 96, 320] : memref<1x3x192x320xf16, [@CMX_NN, 0]> to memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COPY_3:%.*]] = VPUIP.Copy inputs([[INTERP]]#1 : memref<1x3x96x320xf16, [@CMX_NN, 0]>) outputs([[SUBVIEW_3]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) -> memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFF]] : memref<1x3x192x320xf16, [@CMX_NN, 0]>) -> memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x3x192x320xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Maximum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_max.cpp", VPU.kernel_entry = "eltwise_max"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMaximum(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Maximum inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.*]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.*]] = VPUIP.SubView %arg1 [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[MAXIMUM:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Maximum inputs([[SUBVIEW_0]] as %arg2: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as %arg3: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as %arg4: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as %arg5: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as %arg6: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as %arg7: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3, %arg6) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg4, %arg5, %arg7) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MAXIMUM]]#0, [[MAXIMUM]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileAlignCornersInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, ALIGN_CORNERS = 4
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[INBUF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[COPY1]] : memref<1x1x192x320xf16>
}

func.func @TilePytorchHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:  [[IN_BUFF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[COPY0:%.*]]  = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[IN_BUFF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:  [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUT_BUF]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:  return [[COPY1]] : memref<1x1x192x320xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TilePytorchHalfPixelInterpolateWithInitialTileOffsetOnNonScalingDim(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 16, 0], [0, 0, 16, 0]]}(%arg2, %arg3) : memref<1x1x96x160xf16, [@CMX_NN, 0]>, memref<1x1x192x320xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x1x192x320xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    return %4 : memref<1x1x192x320xf16>

    // CHECK:  [[IN_BUFF:%.*]] = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[COPY0:%.*]]  = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs([[IN_BUFF]] : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 47, 0] [1, 1, 49, 160] : memref<1x1x96x160xf16, [@CMX_NN, 0]> to memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 96, 0] [1, 1, 96, 320] : memref<1x1x192x320xf16, [@CMX_NN, 0]> to memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // initial offset c(16) keep value unchanged after tiling
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 16, 0], [0, 0, 16, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 16, 0], [0, 96, 16, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs([[OUT_BUFF]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:  [[OUT_BUF:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:  [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUT_BUF]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:  return [[COPY1]] : memref<1x1x192x320xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "gelu_fp16.cpp", VPU.kernel_entry = "gelu_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGelu(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[GELU:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Gelu inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_HardSigmoid(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_hardsigmoid.cpp", VPU.kernel_entry = "activation_hardsigmoid"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileHardSigmoid(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_HardSigmoid inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[HARDSIGMOID:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_HardSigmoid inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[HARDSIGMOID]]#0, [[HARDSIGMOID]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSoftmax(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>

    // CHECK:   [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[SOFTMAX:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_SoftMax inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSoftmaxWhenAxisIsHighestDim(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 0, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY_CMX]] [0, 0, 32, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT]] [0, 0, 32, 0] [1, 128, 32, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SOFTMAX:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_SoftMax inputs([[SUBVIEW0]] as %arg1: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg3) : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [2]}(%arg2, %arg4) : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x128x32x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPYBACK:%.*]]  = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPYBACK]]  : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileSoftmaxForUnsupportedAxis(%arg0: memref<1x128x1x1xf16>)
        -> memref<1x128x1x1xf16> {
    %0 = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x1x1xf16>) outputs(%0 : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x1x1xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x1x1xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x1x1xf16>) -> memref<1x128x1x1xf16>
    return %4: memref<1x128x1x1xf16>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x1x1xf16>) outputs([[INPUT_CMX]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>

    // CHECK:   [[SOFTMAX:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs([[COPY_CMX]] as %arg1: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[OUTPUT]] as %arg2: memref<1x128x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x1x1xf16, [@CMX_NN, 0]>{
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x1x1xf16>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[SOFTMAX]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x1x1xf16>) -> memref<1x128x1x1xf16>
    // CHECK:   return [[COPYBACK]] : memref<1x128x1x1xf16>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileHalfPixelInterpolateForCMXSizeRequirement(%arg0: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) -> memref<1x16x12x4096xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg2: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x16x12x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x4096xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2048, 6, 16, 1], [4096, 12, 16, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x16x6x2048xf16, [@CMX_NN, 0]>, memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    }
    return %results : memref<1x16x12x4096xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg1: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF]] as %arg2: memref<1x16x12x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x4096xf16, [@CMX_NN, 0]>{
    // CHECK:                         VPUIP.SW.Kernel.run
    // CHECK-NOT:                     VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    return [[INTERP]] : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}
func.func @ConvertOpTest(%arg0: memref<1x64x16x16xf32>) -> memref<1x64x16x16xf16> {
    %0 = memref.alloc() : memref<1x64x16x16xf32, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x64x16x16xf32>) outputs(%0 : memref<1x64x16x16xf32, [@CMX_NN, 0]>) -> memref<1x64x16x16xf32, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x64x16x16xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs(%1 as %arg1: memref<1x64x16x16xf32, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x64x16x16xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x16x16xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x64x16x16xf32, [@CMX_NN, 0]>, memref<1x64x16x16xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x64x16x16xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x64x16x16xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x16x16xf16>) -> memref<1x64x16x16xf16>
    return %4: memref<1x64x16x16xf16>

    // CHECK:    [[MEMREF1:%.*]] = memref.alloc()
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x16x16xf32>)
    // CHECK:    [[MEMREF2:%.*]] = memref.alloc()

    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 32, 16, 16]
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[MEMREF2]] [0, 0, 0, 0] [1, 32, 16, 16]
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[COPY1]] [0, 32, 0, 0] [1, 32, 16, 16]
    // CHECK:    [[SUBVIEW4:%.*]] = VPUIP.SubView [[MEMREF2]] [0, 32, 0, 0] [1, 32, 16, 16]

    // CHECK     [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Convert inputs([[SUBVIEW1]] as %arg1: memref<1x32x16x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg2: memref<1x32x16x16xf32, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW2]] as %arg3: memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>, [[SUBVIEW4]] as %arg4: memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x32x16x16xf16, {order = #NCHW, strides = [16384, 256, 16, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg3)
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg2, %arg4) : memref<1x32x16x16xf32
}

// -----

module @VPU.SW {
  func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "tanh_fp16.cpp", VPU.kernel_entry = "tanh_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileTanh(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Tanh inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[TANH:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Tanh inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[TANH]]#0, [[TANH]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_gather.cpp", VPU.kernel_entry = "single_shave_gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGather(%arg0: memref<30522x26xf16>, %arg1: memref<1x512xsi32>)
        -> memref<1x512x26xf16> {
    %0 = memref.alloc() : memref<30522x26xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<30522x26xf16>) outputs(%0 : memref<30522x26xf16, [@CMX_NN, 0]>) -> memref<30522x26xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x512xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x512xsi32>) outputs(%2 : memref<1x512xsi32, [@CMX_NN, 0]>) -> memref<1x512xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x512x26xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Gather inputs(%1 as %arg2: memref<30522x26xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<1x512xsi32, [@CMX_NN, 0]>) outputs(%4 as %arg4: memref<1x512x26xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x26xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg2, %arg3, %arg4) : memref<30522x26xf16, [@CMX_NN, 0]>, memref<1x512xsi32, [@CMX_NN, 0]>, memref<1x512x26xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x512x26xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x512x26xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x512x26xf16>) -> memref<1x512x26xf16>
    return %6: memref<1x512x26xf16>

    // CHECK:    [[VAR0:%.*]] = memref.alloc() : memref<30522x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<30522x26xf16>) outputs([[VAR0]] : memref<30522x26xf16, [@CMX_NN, 0]>) -> memref<30522x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR2:%.*]] = memref.alloc() : memref<1x512xsi32, [@CMX_NN, 0]>
    // CHECK:    [[VAR3:%.*]] = VPUIP.Copy inputs(%arg1 : memref<1x512xsi32>) outputs([[VAR2]] : memref<1x512xsi32, [@CMX_NN, 0]>) -> memref<1x512xsi32, [@CMX_NN, 0]>
    // CHECK:    [[VAR4:%.*]] = memref.alloc() : memref<1x512x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR5:%.*]] = VPUIP.SubView [[VAR3]] [0, 0] [1, 256] : memref<1x512xsi32, [@CMX_NN, 0]> to memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR6:%.*]] = VPUIP.SubView [[VAR4]] [0, 0, 0] [1, 256, 26] : memref<1x512x26xf16, [@CMX_NN, 0]> to memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR7:%.*]] = VPUIP.SubView [[VAR3]] [0, 256] [1, 256] : memref<1x512xsi32, [@CMX_NN, 0]> to memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR8:%.*]] = VPUIP.SubView [[VAR4]] [0, 256, 0] [1, 256, 26] : memref<1x512x26xf16, [@CMX_NN, 0]> to memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Gather inputs([[VAR1]] as %arg2: memref<30522x26xf16, [@CMX_NN, 0]>, [[VAR5]] as %arg3: memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>, [[VAR1]] as %arg4: memref<30522x26xf16, [@CMX_NN, 0]>, [[VAR7]] as %arg5: memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>) outputs([[VAR6]] as %arg6: memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>, [[VAR8]] as %arg7: memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg2, %arg3, %arg6) : memref<30522x26xf16, [@CMX_NN, 0]>, memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg4, %arg5, %arg7) : memref<30522x26xf16, [@CMX_NN, 0]>, memref<1x256xsi32, {order = #NC, strides = [512, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    [[VAR9:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>, memref<1x256x26xf16, {order = #CHW, strides = [13312, 26, 1]}, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x512x26xf16, [@CMX_NN, 0]>) -> memref<1x512x26xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR10:%.*]] = memref.alloc() : memref<1x512x26xf16>
    // CHECK:    [[VAR11:%.*]] = VPUIP.Copy inputs([[VAR9]] : memref<1x512x26xf16, [@CMX_NN, 0]>) outputs([[VAR10]] : memref<1x512x26xf16>) -> memref<1x512x26xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_gather.cpp", VPU.kernel_entry = "single_shave_gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGatherWithBatchSize2(%arg0: memref<2x64x128xf16>, %arg1: memref<2x32x21xsi32>)
        -> memref<2x32x21x128xf16> {
    %0 = memref.alloc() : memref<2x64x128xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<2x64x128xf16>) outputs(%0 : memref<2x64x128xf16, [@CMX_NN, 0]>) -> memref<2x64x128xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<2x32x21xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<2x32x21xsi32>) outputs(%2 : memref<2x32x21xsi32, [@CMX_NN, 0]>) -> memref<2x32x21xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<2x32x21x128xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Gather inputs(%1 as %arg2: memref<2x64x128xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<2x32x21xsi32, [@CMX_NN, 0]>) outputs(%4 as %arg4: memref<2x32x21x128xf16, [@CMX_NN, 0]>) on tile 0 -> memref<2x32x21x128xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 2]}(%arg2, %arg3, %arg4) : memref<2x64x128xf16, [@CMX_NN, 0]>, memref<2x32x21xsi32, [@CMX_NN, 0]>, memref<2x32x21x128xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<2x32x21x128xf16>
    %6 = VPUIP.Copy inputs(%results : memref<2x32x21x128xf16, [@CMX_NN, 0]>) outputs(%5 : memref<2x32x21x128xf16>) -> memref<2x32x21x128xf16>
    return %6: memref<2x32x21x128xf16>

    // CHECK:    [[VAR0:%.*]] = memref.alloc() : memref<2x64x128xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<2x64x128xf16>) outputs([[VAR0]] : memref<2x64x128xf16, [@CMX_NN, 0]>) -> memref<2x64x128xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR2:%.*]] = memref.alloc() : memref<2x32x21xsi32, [@CMX_NN, 0]>
    // CHECK:    [[VAR3:%.*]] = VPUIP.Copy inputs(%arg1 : memref<2x32x21xsi32>) outputs([[VAR2]] : memref<2x32x21xsi32, [@CMX_NN, 0]>) -> memref<2x32x21xsi32, [@CMX_NN, 0]>
    // CHECK:    [[VAR4:%.*]] = memref.alloc() : memref<2x32x21x128xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR5:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0] [2, 32, 21] : memref<2x64x128xf16, [@CMX_NN, 0]> to memref<2x32x21xf16, {order = #CHW, strides = [8192, 128, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR6:%.*]] = VPUIP.SubView [[VAR3]] [0, 0, 0] [1, 32, 1] : memref<2x32x21xsi32, [@CMX_NN, 0]> to memref<1x32x1xsi32, {order = #CHW, strides = [672, 21, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR7:%.*]] = VPUIP.SubView [[VAR4]] [0, 0, 0, 0] [1, 32, 21, 128] : memref<2x32x21x128xf16, [@CMX_NN, 0]> to memref<1x32x21x128xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR8:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0] [2, 32, 21] : memref<2x64x128xf16, [@CMX_NN, 0]> to memref<2x32x21xf16, {order = #CHW, strides = [8192, 128, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR9:%.*]] = VPUIP.SubView [[VAR3]] [1, 0, 1] [1, 32, 1] : memref<2x32x21xsi32, [@CMX_NN, 0]> to memref<1x32x1xsi32, {order = #CHW, strides = [672, 21, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[VAR10:%.*]] = VPUIP.SubView [[VAR4]] [1, 0, 0, 0] [1, 32, 21, 128] : memref<2x32x21x128xf16, [@CMX_NN, 0]> to memref<1x32x21x128xf16, [@CMX_NN, 0]>

    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Gather inputs([[VAR5]] as %arg2: memref<2x32x21xf16, {order = #CHW, strides = [8192, 128, 1]}, [@CMX_NN, 0]>, [[VAR6]] as %arg3: memref<1x32x1xsi32, {order = #CHW, strides = [672, 21, 1]}, [@CMX_NN, 0]>, [[VAR8]] as %arg4: memref<2x32x21xf16, {order = #CHW, strides = [8192, 128, 1]}, [@CMX_NN, 0]>, [[VAR9]] as %arg5: memref<1x32x1xsi32, {order = #CHW, strides = [672, 21, 1]}, [@CMX_NN, 0]>) outputs([[VAR7]] as %arg6: memref<1x32x21x128xf16, [@CMX_NN, 0]>, [[VAR10]] as %arg7: memref<1x32x21x128xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x32x21x128xf16, [@CMX_NN, 0]>, memref<1x32x21x128xf16, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [2, 2]}(%arg2, %arg3, %arg6) : memref<2x32x21xf16, {order = #CHW, strides = [8192, 128, 1]}, [@CMX_NN, 0]>, memref<1x32x1xsi32, {order = #CHW, strides = [672, 21, 1]}, [@CMX_NN, 0]>, memref<1x32x21x128xf16, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [2, 2]}(%arg4, %arg5, %arg7) : memref<2x32x21xf16, {order = #CHW, strides = [8192, 128, 1]}, [@CMX_NN, 0]>, memref<1x32x1xsi32, {order = #CHW, strides = [672, 21, 1]}, [@CMX_NN, 0]>, memref<1x32x21x128xf16, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[VAR11:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x32x21x128xf16, [@CMX_NN, 0]>, memref<1x32x21x128xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<2x32x21x128xf16, [@CMX_NN, 0]>) -> memref<2x32x21x128xf16, [@CMX_NN, 0]>
    // CHECK:    [[VAR12:%.*]] = memref.alloc() : memref<2x32x21x128xf16>
    // CHECK:    [[VAR13:%.*]] = VPUIP.Copy inputs([[VAR11]] : memref<2x32x21x128xf16, [@CMX_NN, 0]>) outputs([[VAR12]] : memref<2x32x21x128xf16>) -> memref<2x32x21x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_Sigmoid(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "sigmoid_fp16.cpp", VPU.kernel_entry = "sigmoid_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSigmoid(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Sigmoid inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 64, 0, 0] [1, 64, 64, 32] : memref<1x128x64x32xf16, [@CMX_NN, 0]> to memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Sigmoid inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_DepthToSpace(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSWDepthToSpace(%arg0: memref<1x128x12x270xf16, #NHWC>)
        -> memref<1x8x48x1080xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs(%0 : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%1 as %arg1: memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg1, %arg2) : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    return %4: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs([[INBUF]] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 128, 6, 270] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]> to memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 8, 24, 1080] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]> to memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 6, 0] [1, 128, 6, 270] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]> to memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 24, 0] [1, 8, 24, 1080] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]> to memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:    [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_DepthToSpace inputs([[IN_SUBVIEW0]] as %arg1: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg1, %arg3) : memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg2, %arg4) : memref<1x128x6x270xf16, {order = #NHWC, strides = [414720, 1, 34560, 128]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 : memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>, memref<1x8x24x1080xf16, {order = #NHWC, strides = [414720, 1, 8640, 8]}, [@CMX_NN, 0]>) outputs([[OUTBUF]] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    return [[COPY1]] : memref<1x8x48x1080xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_DepthToSpace(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileDMADepthToSpace(%arg0: memref<1x128x12x270xf16, #NHWC>)
        -> memref<1x8x48x1080xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs(%0 : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_DepthToSpace inputs(%1 as %arg1: memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, 0]}(%arg1, %arg2) : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    return %4: memref<1x8x48x1080xf16, #NHWC>

    // CHECK:    [[INBUF:%.*]] = memref.alloc() : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x12x270xf16, #NHWC>) outputs([[INBUF]] : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[RES:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_DepthToSpace inputs([[COPY0]] as %arg1: memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF]] as %arg2: memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [4, 0]}(%arg1, %arg2) : memref<1x128x12x270xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[RES]] : memref<1x8x48x1080xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x8x48x1080xf16, #NHWC>) -> memref<1x8x48x1080xf16, #NHWC>
    // CHECK:    return [[COPY1]] : memref<1x8x48x1080xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Clamp(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "clamp_fp16.cpp", VPU.kernel_entry = "clamp_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileClamp(%arg0: memref<1x5x34x60xf16, #NHWC>)
        -> memref<1x5x34x60xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs(%0 : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Clamp inputs(%1 as %arg1: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg1, %arg2) : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>, memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
      }
    %3 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    return %4: memref<1x5x34x60xf16, #NHWC>

    // CHECK:   [[INBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[INCOPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs([[INBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB0:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB1:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Clamp
    // CHECK-SAME:    inputs([[IN_SUB0]] as %arg1: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[IN_SUB1]] as %arg2: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUT_SUB0]] as %arg3: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[OUT_SUB1]] as %arg4: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:    -> (memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg1, %arg3) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg2, %arg4) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 :
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUTBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_DDR:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:   [[COPY:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUT_DDR]] : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_Abs(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_abs.cpp", VPU.kernel_entry = "activation_abs"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileAbs(%arg0: memref<1x5x34x60xf16, #NHWC>)
        -> memref<1x5x34x60xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs(%0 : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Abs inputs(%1 as %arg1: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run (%arg1, %arg2) : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>, memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
      }
    %3 = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
    return %4: memref<1x5x34x60xf16, #NHWC>

    // CHECK:   [[INBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[INCOPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x5x34x60xf16, #NHWC>) outputs([[INBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUTBUF:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB0:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB0:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 0, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SUB1:%.*]] = VPUIP.SubView [[INCOPY]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SUB1:%.*]] = VPUIP.SubView [[OUTBUF]] [0, 0, 17, 0] [1, 5, 17, 60] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]> to memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   [[RES:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Abs
    // CHECK-SAME:    inputs([[IN_SUB0]] as %arg1: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[IN_SUB1]] as %arg2: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUT_SUB0]] as %arg3: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[OUT_SUB1]] as %arg4: memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:    -> (memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>, memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[RES]]#0, [[RES]]#1 :
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:    memref<1x5x17x60xf16, {order = #NHWC, strides = [10200, 1, 300, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[OUTBUF]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_DDR:%.*]] = memref.alloc() : memref<1x5x34x60xf16, #NHWC>
    // CHECK:   [[COPY:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x5x34x60xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUT_DDR]] : memref<1x5x34x60xf16, #NHWC>) -> memref<1x5x34x60xf16, #NHWC>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "single_shave_topk.cpp", VPU.kernel_entry = "single_shave_topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileTopKSubViewWithOneDPUGroup(%arg0: memref<1x1x10x1xf16, #NCWH>)
        -> (memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>) {
    %0 = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x10x1xf16, #NCWH>) outputs(%0 : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg10, %arg11, %arg12) : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x10x1xf16, #NCWH>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x10x1xf16, #NCWH>) -> memref<1x1x10x1xf16, #NCWH>
    %7 = memref.alloc() : memref<1x1x10x1xsi32, #NCWH>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x10x1xsi32, #NCWH>) -> memref<1x1x10x1xsi32, #NCWH>
    return %6, %8 : memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>


    // CHECK: [[IN_BUF:%.*]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[IN_COPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x10x1xf16, #NCWH>) outputs([[IN_BUF]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[OUT_BUF_0:%.*]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[OUT_BUF_1:%.*]] = memref.alloc() : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>

    // CHECK: [[IN_SUBVIEW_0:%.*]] = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUF_0]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUF_1]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[IN_SUBVIEW_1:%.*]] = VPUIP.SubView [[IN_COPY]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_2:%.*]] = VPUIP.SubView [[OUT_BUF_0]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_3:%.*]] = VPUIP.SubView [[OUT_BUF_1]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[TOPK:%.*]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0>} @VPU.SW::@builtin_TopK inputs([[IN_SUBVIEW_0]] as %arg1: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[IN_SUBVIEW_1]] as %arg2: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW_0]] as %arg3: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_1]] as %arg4: memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_2]] as %arg5: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_3]] as %arg6: memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg1, %arg3, %arg4) : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg2, %arg5, %arg6) : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_0:%.*]] = VPUIP.ConcatView inputs([[TOPK]]#0, [[TOPK]]#2 : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[OUT_BUF_0]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_1:%.*]] = VPUIP.ConcatView inputs([[TOPK]]#1, [[TOPK]]#3 : memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[OUT_BUF_1]] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>

    // CHECK: [[OUT_DDR_BUF_0:%.*]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH>
    // CHECK: [[OUT_DDR_COPY_0:%.*]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs([[OUT_DDR_BUF_0]] : memref<1x1x10x1xf16, #NCWH>) -> memref<1x1x10x1xf16, #NCWH>
    // CHECK: [[OUT_DDR_BUF_1:%.*]] = memref.alloc() : memref<1x1x10x1xsi32, #NCWH>
    // CHECK: [[OUT_DDR_COPY_1:%.*]] = VPUIP.Copy inputs([[CONCAT_1]] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) outputs([[OUT_DDR_BUF_1]] : memref<1x1x10x1xsi32, #NCWH>) -> memref<1x1x10x1xsi32, #NCWH>
    // CHECK: return [[OUT_DDR_COPY_0]], [[OUT_DDR_COPY_1]] : memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>
}

// -----

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "single_shave_topk.cpp", VPU.kernel_entry = "single_shave_topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileTopKCopyWithOneDPUGroup(%arg0: memref<1x8x16x16xf16>)
        -> (memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>) {
    %0 = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16>) outputs(%0 : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x16x16xf16, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x16x16xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg10, %arg11, %arg12) : memref<1x8x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x16x16xf16>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x16x16xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x16x16xf16>) -> memref<1x1x16x16xf16>
    %7 = memref.alloc() : memref<1x1x16x16xsi32>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x16x16xsi32>) -> memref<1x1x16x16xsi32>
    return %6, %8 : memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>

    // CHECK: [[IN_BUF:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_COPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16>) outputs([[IN_BUF]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x8x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1_OUT:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1_COPY:%.*]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_1_OUT]] : memref<1x1x8x16xf16, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, [@CMX_NN, 0]>

    // CHECK: [[RUN_1_OUTPUT_1:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[RUN_1_OUTPUT_2:%.*]] = memref.alloc() : memref<1x1x8x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_2:%.*]] = VPUIP.SubView [[IN_COPY]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x8x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_2_OUT:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_2_COPY:%.*]]  = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2_OUT]] : memref<1x1x8x16xf16, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, [@CMX_NN, 0]>

    // CHECK: [[RUN_2_OUTPUT_1:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[RUN_2_OUTPUT_2:%.*]] = memref.alloc() : memref<1x1x8x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[TOPK:%.*]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0>} @VPU.SW::@builtin_TopK inputs([[SUBVIEW_1_COPY]] as %arg1: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[SUBVIEW_2_COPY]] as %arg2: memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[RUN_1_OUTPUT_1]] as %arg3: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[RUN_1_OUTPUT_2]] as %arg4: memref<1x1x8x16xsi32, [@CMX_NN, 0]>, [[RUN_2_OUTPUT_1]] as %arg5: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[RUN_2_OUTPUT_2]] as %arg6: memref<1x1x8x16xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg1, %arg3, %arg4) : memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg2, %arg5, %arg6) : memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>
    // CHECK:  }

    // CHECK: [[CONCAT_1_OUTPUT:%.*]] = memref.alloc() : memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_1:%.*]] = VPUIP.SubView [[CONCAT_1_OUTPUT]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x1x16x16xsi32, [@CMX_NN, 0]> to memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_1_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#1 : memref<1x1x8x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_1_1]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_SUBVIEW_1_2:%.*]] = VPUIP.SubView [[CONCAT_1_OUTPUT]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x1x16x16xsi32, [@CMX_NN, 0]> to memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_2_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#3 : memref<1x1x8x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_1_2]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_1:%.*]] = VPUIP.ConcatView inputs([[OUTPUT_SUBVIEW_1_1_CP]], [[OUTPUT_SUBVIEW_1_2_CP]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[CONCAT_1_OUTPUT]] : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) -> memref<1x1x16x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_2_OUTPUT:%.*]]  = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_1:%.*]] = VPUIP.SubView [[CONCAT_2_OUTPUT]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x1x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_1_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#0 : memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_2_1]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_2:%.*]] = VPUIP.SubView [[CONCAT_2_OUTPUT]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x1x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_2_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#2 : memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_2_2]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_2:%.*]] = VPUIP.ConcatView inputs([[OUTPUT_SUBVIEW_2_1_CP]], [[OUTPUT_SUBVIEW_2_2_CP]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[CONCAT_2_OUTPUT]] : memref<1x1x16x16xf16, [@CMX_NN, 0]>) -> memref<1x1x16x16xf16, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_2_DDR:%.*]] = memref.alloc() : memref<1x1x16x16xf16>
    // CHECK: [[CONCAT_2_DDR_COPY:%.*]] = VPUIP.Copy inputs([[CONCAT_2]] : memref<1x1x16x16xf16, [@CMX_NN, 0]>) outputs([[CONCAT_2_DDR]] : memref<1x1x16x16xf16>) -> memref<1x1x16x16xf16>
    // CHECK: [[CONCAT_1_DDR:%.*]]  = memref.alloc() : memref<1x1x16x16xsi32>
    // CHECK: [[CONCAT_1_DDR_COPY:%.*]] = VPUIP.Copy inputs([[CONCAT_1]] : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) outputs([[CONCAT_1_DDR]] : memref<1x1x16x16xsi32>) -> memref<1x1x16x16xsi32>
    // CHECK: return [[CONCAT_2_DDR_COPY]], [[CONCAT_1_DDR_COPY]] : memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>
}

// -----

module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMinimum(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.*]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.*]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[MINIMUM:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_Minimum inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MINIMUM]]#0, [[MINIMUM]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}
