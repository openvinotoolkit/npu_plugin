//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --tile-act-shave-kernel-task --canonicalize %s | FileCheck %s

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
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%1 as %arg1: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>{
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
    // CHECK:   [[MVN:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NWHC, strides = [262144, 1, 128, 8192]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16, #NWHC>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, #NWHC, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16, #NWHC>) -> memref<1x128x64x32xf16, #NWHC>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16, #NWHC>
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
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
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
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                           VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
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
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg2: memref<1x3x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x3x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x192x320xf16, [@CMX_NN, 0]>{
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
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs([[COPY_0]] as %arg1: memref<1x3x49x160xf16, [@CMX_NN, 0]>, [[COPY_1]] as %arg2: memref<1x3x49x160xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFF_0]] as %arg3: memref<1x3x96x320xf16, [@CMX_NN, 0]>, [[OUTPUT_BUFF_1]] as %arg4: memref<1x3x96x320xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x96x320xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x3x49x160xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 3, 1], [320, 192, 3, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x3x49x160xf16, [@CMX_NN, 0]>, memref<1x3x96x320xf16, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[OUTPUT_BUFF:%.*]] = memref.alloc() : memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 3, 96, 320] : memref<1x3x192x320xf16, [@CMX_NN, 0]> to memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COPY_2:%.*]] = VPUIP.Copy inputs([[INTERP]]#0 : memref<1x3x96x320xf16, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) -> memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 96, 0] [1, 3, 96, 320] : memref<1x3x192x320xf16, [@CMX_NN, 0]> to memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COPY_3:%.*]] = VPUIP.Copy inputs([[INTERP]]#1 : memref<1x3x96x320xf16, [@CMX_NN, 0]>) outputs([[SUBVIEW_3]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) -> memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x3x96x320xf16, {order = #NCHW, strides = [184320, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs(%8 : memref<1x3x192x320xf16, [@CMX_NN, 0]>) -> memref<1x3x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x3x192x320xf16, [@CMX_NN, 0]>
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
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
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
    // CHECK:    [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x1x192x320xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x192x320xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x1x192x320xf16>) -> memref<1x1x192x320xf16>
    // CHECK:    return [[COPY1]] : memref<1x1x192x320xf16>
}

func.func @TilePytorchHalfPixelInterpolate(%arg0: memref<1x1x96x160xf16>) -> memref<1x1x192x320xf16> {
    %0 = memref.alloc() : memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x96x160xf16>) outputs(%0 : memref<1x1x96x160xf16, [@CMX_NN, 0]>) -> memref<1x1x96x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x192x320xf16, [@CMX_NN, 0]>
    // LINEAR_ONNX = 2, PYTORCH_HALF_PIXEL = 1
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
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
    // CHECK:  [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 0, 0], [0, 96, 0, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
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
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg2: memref<1x1x96x160xf16, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x1x192x320xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x192x320xf16, [@CMX_NN, 0]>{
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
    // CHECK:  [[INTERP:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs([[IN_SUBVIEW0]] as %arg1: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>){
    // initial offset c(16) keep value unchanged after tiling
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 0, 16, 0], [0, 0, 16, 0]]}(%arg1, %arg3) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [2, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [160, 96, 1, 1], [320, 192, 1, 1], [2, 3], -7.500000e-01, [0, 47, 16, 0], [0, 96, 16, 0]]}(%arg2, %arg4) : memref<1x1x49x160xf16, {order = #NCHW, strides = [15360, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[INTERP]]#0, [[INTERP]]#1 : memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>, memref<1x1x96x320xf16, {order = #NCHW, strides = [61440, 61440, 320, 1]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x192x320xf16, [@CMX_NN, 0]>) -> memref<1x1x192x320xf16, [@CMX_NN, 0]>
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
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Gelu inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
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
    // CHECK:    [[GELU:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_Gelu inputs([[IN_SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[IN_SUBVIEW1]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW1]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg3) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg4) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[GELU]]#0, [[GELU]]#1 : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTBUF_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTBUF_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSoftmax(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
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
    // CHECK:   [[SOFTMAX:%.*]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[SUBVIEW0]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW1]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, [@CMX_NN, 0]>){
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
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileSoftmaxForUnsupportedAxis(%arg0: memref<1x128x64x32xf16>)
        -> memref<1x128x64x32xf16> {
    %0 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs(%0 : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs(%1 as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x128x64x32xf16>
    %4 = VPUIP.Copy inputs(%results : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    return %4: memref<1x128x64x32xf16>

    // CHECK:   [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY_CMX:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) -> memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   [[OUTPUT:%.*]] = memref.alloc() : memref<1x128x64x32xf16, [@CMX_NN, 0]>

    // CHECK:   [[SOFTMAX:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[COPY_CMX]] as %arg1: memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT]] as %arg2: memref<1x128x64x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x64x32xf16, [@CMX_NN, 0]>{
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [2]}(%arg1, %arg2) : memref<1x128x64x32xf16, [@CMX_NN, 0]>, memref<1x128x64x32xf16, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:   [[COPYBACK:%.*]] = VPUIP.Copy inputs([[SOFTMAX]] : memref<1x128x64x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:   return [[COPYBACK]] : memref<1x128x64x32xf16>
}

// -----

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "single_shave_interpolate.cpp", VPU.kernel_entry = "singleShaveInterpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileHalfPixelInterpolateForCMXSizeRequirement(%arg0: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) -> memref<1x16x12x4096xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg2: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x16x12x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x4096xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [2, 0, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2048, 6, 16, 1], [4096, 12, 16, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg2, %arg3) : memref<1x16x6x2048xf16, [@CMX_NN, 0]>, memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    }
    return %results : memref<1x16x12x4096xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
    // CHECK:    [[INTERP:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Interpolate inputs(%arg0 as %arg1: memref<1x16x6x2048xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF]] as %arg2: memref<1x16x12x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x12x4096xf16, [@CMX_NN, 0]>{
    // CHECK:                         VPUIP.SW.Kernel.run
    // CHECK-NOT:                     VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    return [[INTERP]] : memref<1x16x12x4096xf16, [@CMX_NN, 0]>
}
