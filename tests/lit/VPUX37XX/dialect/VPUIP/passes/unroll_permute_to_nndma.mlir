//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --unroll-permute-to-nndma  %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNHWCToNCHW
func.func @PermuteToDMAWithNHWCToNCHW() -> memref<1x8x16x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 512 : i64, dstWidth = 2 : i64, len = 16 : i64, numPlanes = 256 : i64, srcPlaneStride = 16 : i64, srcStride = 2 : i64, srcWidth = 16 : i64}
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWC
func.func @PermuteToDMAWithNCHWToNHWC() -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<128x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<16x128xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 256 : i64, dstWidth = 2 : i64, len = 32 : i64, numPlanes = 128 : i64, srcPlaneStride = 32 : i64, srcStride = 2 : i64, srcWidth = 32 : i64}
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<128x16xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<16x128xf16, [@CMX_NN, 0]>) -> memref<16x128xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAFromTranspose
func.func @PermuteToDMAFromTranspose() -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<32x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<8x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 16 : i64, numPlanes = 32 : i64, srcPlaneStride = 16 : i64, srcStride = 2 : i64, srcWidth = 16 : i64}
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<32x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<8x32xf16, [@CMX_NN, 0]>) -> memref<8x32xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithLargePlaneNumber
func.func @PermuteToDMAWithLargePlaneNumber() -> memref<1x8x32x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x32x16xf16, [@CMX_NN, 0]>) -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x8x32x16xf16, [@CMX_NN, 0]>


    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4608> -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 1024 : i64, dstWidth = 2 : i64, len = 16 : i64, numPlanes = 256 : i64, srcPlaneStride = 16 : i64, srcStride = 2 : i64, srcWidth = 16 : i64}
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER0]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER0]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 1024 : i64, dstWidth = 2 : i64, len = 16 : i64, numPlanes = 256 : i64, srcPlaneStride = 16 : i64, srcStride = 2 : i64, srcWidth = 16 : i64}
    //CHECK-SAME:       port = 1 : i64}
    //CHECK:            inputs([[INPUT_BUFFER1]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER1]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x32x16xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @UnrollDistributedPermuteDMA
func.func @UnrollDistributedPermuteDMA() -> memref<1x3x24x24xf16, #NHWC, @DDR> {
    %result = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x3x24x24xf16, #NHWC, @DDR>
    %cst = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %12 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x3x24x24xui8, @DDR>
    %13 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x3x12x24xf16, #NHWC, @DDR>
    %14 = VPURT.DeclareBuffer "NetworkOutput" [0] <1728> -> memref<1x3x12x24xf16, #NHWC, @DDR>
    %15 = VPURT.DeclareBuffer "CMX_NN" [0] <3456> -> memref<1x3x24x24xui8, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x3x24x24xf16, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x3x24x24xf16, @DDR>
    %18 = VPURT.DeclareBuffer "DDR" <3456> -> memref<1x16x24x24xf16, @DDR>
    %19 = VPURT.DeclareBuffer "CMX_NN" <5440> -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %20 = VPURT.DeclareBuffer "CMX_NN" [0] <5440> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer "CMX_NN" [1] <5440> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer "CMX_NN" [0] <5184> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer "CMX_NN" [1] <5184> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    %24 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <5184> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %25 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %26 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 1]>
    %27 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %28 = VPURT.DeclareBuffer "CMX_NN" <14656> -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %29 = VPURT.DeclareBuffer "CMX_NN" [0] <14656> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer "CMX_NN" [1] <14656> -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer "DDR" <3456> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %32 = VPURT.DeclareBuffer "DDR" <6912> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %33 = VPURT.DeclareBuffer "DDR" <10368> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %34 = VPURT.DeclareBuffer "DDR" <13824> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %35 = VPURT.DeclareBuffer "DDR" <17280> -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %36 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x24x24xf16, {order = #NCHW, strides = [1728, 576, 24, 1]}, @DDR>
    %37 = VPURT.DeclareBuffer "DDR" <20736> -> memref<1x1x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    %38 = VPURT.DeclareBuffer "CMX_NN" [0] <14656> -> memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>
    %39 = VPURT.DeclareBuffer "CMX_NN" [1] <14656> -> memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 1]>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {cycleBegin = 0 : i64, cycleEnd = 72 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%12 : memref<1x3x24x24xui8, @DDR>) outputs(%15 : memref<1x3x24x24xui8, [@CMX_NN, 0]>) -> memref<1x3x24x24xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 72 : i64, cycleEnd = 78 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<16x1x1x4xsi32>) outputs(%24 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 72 : i64, cycleEnd = 74 : i64, isTrailingSWLayer = false} {
      %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%15 as %arg2: memref<1x3x24x24xui8, [@CMX_NN, 0]>) outputs(%16 as %arg3: memref<1x3x24x24xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x24x24xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x3x24x24xui8, [@CMX_NN, 0]>, memref<1x3x24x24xf16, [@CMX_NN, 0]>
      }
    }

    // expand input
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) attributes {cycleBegin = 78 : i64, cycleEnd = 222 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%16 : memref<1x3x24x24xf16, [@CMX_NN, 0]>) outputs(%17 : memref<1x3x24x24xf16, @DDR>) -> memref<1x3x24x24xf16, @DDR>
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {cycleBegin = 222 : i64, cycleEnd = 366 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%31 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 366 : i64, cycleEnd = 510 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%32 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 510 : i64, cycleEnd = 654 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%33 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {cycleBegin = 654 : i64, cycleEnd = 798 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%34 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 798 : i64, cycleEnd = 942 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1x3x24x24xf16, @DDR>) outputs(%35 : memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x3x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 942 : i64, cycleEnd = 990 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%36 : memref<1x1x24x24xf16, {order = #NCHW, strides = [1728, 576, 24, 1]}, @DDR>) outputs(%37 : memref<1x1x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>) -> memref<1x1x24x24xf16, {order = #NCHW, strides = [9216, 576, 24, 1]}, @DDR>
    }
    // permute
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NCEClusterTiling inputs(%18 as %arg2: memref<1x16x24x24xf16>) outputs(%19 as %arg3: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %41 = VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64, src_plane_stride = 0 : i64} inputs(%arg2 : memref<1x16x24x24xf16>) outputs(%arg3 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
      }
    }

    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 1758 : i64, cycleEnd = 1759 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x16xui8>) outputs(%27 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }

    // NCE task
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {cycleBegin = 1759 : i64, cycleEnd = 1967 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "MAXPOOL"} input(%20 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%22 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) activation_window(%25 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%28 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%29 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {cycleBegin = 1759 : i64, cycleEnd = 1967 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "MAXPOOL"} input(%21 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%23 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) activation_window(%26 : memref<1x1x1x16xui8, [@CMX_NN, 1]>) parent_input(%19 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%28 : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%30 : memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x12x24xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 12, 0]}
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }
    // copy result
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {cycleBegin = 1967 : i64, cycleEnd = 2111 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 0 : i64} inputs(%38 : memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>) outputs(%13 : memref<1x3x12x24xf16, #NHWC, @DDR>) -> memref<1x3x12x24xf16, #NHWC, @DDR>
    }
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {cycleBegin = 1967 : i64, cycleEnd = 2111 : i64, isTrailingSWLayer = false} {
      %40 = VPUIP.NNDMA {port = 1 : i64} inputs(%39 : memref<1x3x12x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 1]>) outputs(%14 : memref<1x3x12x24xf16, #NHWC, @DDR>) -> memref<1x3x12x24xf16, #NHWC, @DDR>
    }
    return %result : memref<1x3x24x24xf16, #NHWC, @DDR>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR7:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR8:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR9:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR10:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR11:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUF_0:%.*]] = VPURT.DeclareBuffer "DDR" <3456> -> memref<16x288xf16, @DDR>
    //CHECK:    [[INPUT_BUF_1:%.*]] = VPURT.DeclareBuffer "DDR" <4032> -> memref<16x288xf16, @DDR>
    //CHECK:    [[CMX_BUF_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <5440> -> memref<288x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[CMX_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <5440> -> memref<288x16xf16, [@CMX_NN, 1]>



    //CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR7]] : !VPURT.Barrier) updates([[BAR8]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR8]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:     dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 576 : i64, numPlanes = 16 : i64, srcPlaneStride = 1152 : i64, srcStride = 2 : i64, srcWidth = 576 : i64}
    //CHECK-SAME:     port = 0 : i64}
    //CHECK-SAME:     inputs([[INPUT_BUF_0]] : memref<16x288xf16, @DDR>) outputs([[CMX_BUF_0]] : memref<288x16xf16, [@CMX_NN, 0]>) -> memref<288x16xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR8]] : !VPURT.Barrier) updates([[BAR9]] : !VPURT.Barrier)
    //CHECK:      VPUIP.PermuteDMA {
    //CHECK-SAME:     dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 576 : i64, numPlanes = 16 : i64, srcPlaneStride = 1152 : i64, srcStride = 2 : i64, srcWidth = 576 : i64}
    //CHECK-SAME:     port = 1 : i64}
    //CHECK-SAME:     inputs([[INPUT_BUF_1]] : memref<16x288xf16, @DDR>) outputs([[CMX_BUF_1]] : memref<288x16xf16, [@CMX_NN, 1]>) -> memref<288x16xf16, [@CMX_NN, 1]>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR9]] : !VPURT.Barrier) updates([[BAR10]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x16xui8>) outputs(%31 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    }

    // nce task
    //CHECK:    VPURT.Task waits([[BAR10]] : !VPURT.Barrier) updates([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR10]] : !VPURT.Barrier) updates([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:    }

    // copy back
    //CHECK:    VPURT.Task waits([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA {port = 0 : i64}
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR11]] : !VPURT.Barrier)
    //CHECK:      VPUIP.NNDMA {port = 1 : i64}
    //CHECK:    }
    //CHECK:    return %0 : memref<1x3x24x24xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>


// CHECK-LABEL: @PermuteToDMAWithHWCToWHC
func.func @PermuteToDMAWithHWCToWHC() -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = #map1, port = 0 : i64}
                inputs(%input: memref<1x16x4x76xf16, #map0, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %output: memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[VAR0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16x4x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[VAR2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <9728> -> memref<4x16x76xf16, [@CMX_NN, 0]>
    //CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:         VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 152 : i64, dstStride = 2432 : i64, dstWidth = 152 : i64, len = 608 : i64, numPlanes = 16 : i64, srcPlaneStride = 608 : i64, srcStride = 2 : i64, srcWidth = 608 : i64}, port = 0 : i64}
    //CHECK-SAME:    inputs([[VAR0]] : memref<16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<4x16x76xf16, [@CMX_NN, 0]>) -> memref<4x16x76xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[VAR1]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-LABEL: @PermuteToDMAWithHWCToHCW
func.func @PermuteToDMAWithHWCToHCW() -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {mem_perm = #map, port = 0 : i64}
                inputs(%input: memref<1x16x4x76xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %output: memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>


    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[VAR0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16x4x76xf16, [@CMX_NN, 0]>
    //CHECK:    [[VAR1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <9728> -> memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[VAR2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <9728> -> memref<16x76x4xf16, [@CMX_NN, 0]>
    //CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 8 : i64, dstWidth = 2 : i64, len = 2432 : i64, numPlanes = 4 : i64, srcPlaneStride = 152 : i64, srcStride = 608 : i64, srcWidth = 152 : i64}, port = 0 : i64}
    //CHECK-SAME: inputs([[VAR0]] : memref<16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<16x76x4xf16, [@CMX_NN, 0]>) -> memref<16x76x4xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[VAR1]] : memref<1x16x4x76xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteSEGMENTED
func.func @ClusterPermuteDMAWithExpandAndPermuteSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x3x32x32x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x3x32x32x!qElemType>) outputs(%1 as %arg1: memref<1x16x32x32x!qElemType, #NHWC, @CMX_NN>) -> !OutputDistributed {
        VPUIP.PermuteDMA {mem_perm = #map, port = 0 : i64}
              inputs(%arg0 : memref<1x3x32x32x!qElemType>)
              outputs(%arg1 : memref<1x16x32x32x!qElemType, #NHWC, @CMX_NN>) -> memref<1x16x32x32x!qElemType, #NHWC, @CMX_NN>
      }
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<3x512x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer "DDR" <512> -> memref<3x512x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<512x16x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 1 : i64, dstStride = 16 : i64, dstWidth = 1 : i64, len = 512 : i64, numPlanes = 3 : i64, srcPlaneStride = 1024 : i64, srcStride = 1 : i64, srcWidth = 512 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<3x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<512x16x!qElemType, [@CMX_NN, 0]>) -> memref<512x16x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 1 : i64, dstStride = 16 : i64, dstWidth = 1 : i64, len = 512 : i64, numPlanes = 3 : i64, srcPlaneStride = 1024 : i64, srcStride = 1 : i64, srcWidth = 512 : i64}, port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<3x512x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<512x16x!qElemType, [@CMX_NN, 1]>) -> memref<512x16x!qElemType, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
    strides = [2, 2],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithExpandAndPermuteOVERLAPPED
func.func @ClusterPermuteDMAWithExpandAndPermuteOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x3x224x224x!qElemType, @DDR>
    %1 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x3x224x224x!qElemType>) outputs(%1 as %arg1: memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>) -> !OutputDistributed {
        VPUIP.PermuteDMA {mem_perm = #map, port = 0 : i64}
              inputs(%arg0 : memref<1x3x224x224x!qElemType>)
              outputs(%arg1 : memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>) -> memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>
      }
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<3x25536x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer "DDR" <24416> -> memref<3x25760x!qElemType, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, strides = [2, 2], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<25536x4x!qElemType, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 1 : i64, dstStride = 4 : i64, dstWidth = 1 : i64, len = 25536 : i64, numPlanes = 3 : i64, srcPlaneStride = 50176 : i64, srcStride = 1 : i64, srcWidth = 25536 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<3x25536x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<25536x4x!qElemType, [@CMX_NN, 0]>) -> memref<25536x4x!qElemType, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 1 : i64, dstStride = 4 : i64, dstWidth = 1 : i64, len = 25760 : i64, numPlanes = 3 : i64, srcPlaneStride = 50176 : i64, srcStride = 1 : i64, srcWidth = 25760 : i64}, port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<3x25760x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<25760x4x!qElemType, [@CMX_NN, 1]>) -> memref<25760x4x!qElemType, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [7, 7], pads = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, strides = [2, 2], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWCForNetworkOutput
func.func @PermuteToDMAWithNCHWToNHWCForNetworkOutput() -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkOutput" [4] <0> -> memref<1x32x14x7xf16, @DDR>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
      VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64}
            inputs(%input : memref<1x32x14x7xf16, @DDR>)
            outputs(%output : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
   }

    return %output: memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [4] <0> -> memref<32x98xf16, @DDR>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <6272> -> memref<98x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 196 : i64, numPlanes = 32 : i64, srcPlaneStride = 196 : i64, srcStride = 2 : i64, srcWidth = 196 : i64}
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<32x98xf16, @DDR>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<98x32xf16, [@CMX_NN, 0]>) -> memref<98x32xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWCForNetworkInput
func.func @PermuteToDMAWithNCHWToNHWCForNetworkInput() -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkInput" [2] <0> -> memref<1x32x14x7xf16, @DDR>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier)  {
      VPUIP.PermuteDMA {mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, port = 0 : i64}
            inputs(%input : memref<1x32x14x7xf16, @DDR>)
            outputs(%output : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
   }

    return %output: memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "NetworkInput" [2] <0> -> memref<32x98xf16, @DDR>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <6272> -> memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <6272> -> memref<98x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 196 : i64, numPlanes = 32 : i64, srcPlaneStride = 196 : i64, srcStride = 2 : i64, srcWidth = 196 : i64}
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<32x98xf16, @DDR>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<98x32xf16, [@CMX_NN, 0]>) -> memref<98x32xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x32x14x7xf16, #NHWC, [@CMX_NN, 0]>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x14x7xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteToDMAWithNCHWToNHWCForNetworkOutput
func.func @ClusterPermuteToDMAWithNCHWToNHWCForNetworkOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x32x14x7xf16, @DDR>
    %1 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x32x14x7xf16>) outputs(%1 as %arg1: memref<1x32x14x7xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64}
              inputs(%arg0 : memref<1x32x14x7xf16>)
              outputs(%arg1 : memref<1x32x14x7xf16, #NHWC, @CMX_NN>) -> memref<1x32x14x7xf16, #NHWC, @CMX_NN>
      }
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<32x49xf16, [@DDR, 0]>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <98> -> memref<32x49xf16, [@DDR, 0]>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<49x32xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<49x32xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 98 : i64, numPlanes = 32 : i64, srcPlaneStride = 196 : i64, srcStride = 2 : i64, srcWidth = 98 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<32x49xf16, [@DDR, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<49x32xf16, [@CMX_NN, 0]>) -> memref<49x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 98 : i64, numPlanes = 32 : i64, srcPlaneStride = 196 : i64, srcStride = 2 : i64, srcWidth = 98 : i64}, port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<32x49xf16, [@DDR, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<49x32xf16, [@CMX_NN, 1]>) -> memref<49x32xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x14x7xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @ClusterPermuteToDMAWithNCHWToNHWCForNetworkInput
func.func @ClusterPermuteToDMAWithNCHWToNHWCForNetworkInput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x32x14x7xf16, @DDR>
    %1 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x32x14x7xf16>) outputs(%1 as %arg1: memref<1x32x14x7xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64}
              inputs(%arg0 : memref<1x32x14x7xf16>)
              outputs(%arg1 : memref<1x32x14x7xf16, #NHWC, @CMX_NN>) -> memref<1x32x14x7xf16, #NHWC, @CMX_NN>
      }
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<32x49xf16, [@DDR, 0]>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <98> -> memref<32x49xf16, [@DDR, 0]>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<49x32xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<49x32xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 98 : i64, numPlanes = 32 : i64, srcPlaneStride = 196 : i64, srcStride = 2 : i64, srcWidth = 98 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<32x49xf16, [@DDR, 0]>)
    //CHECK:                outputs([[OUTPUT0]] : memref<49x32xf16, [@CMX_NN, 0]>) -> memref<49x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 64 : i64, dstWidth = 2 : i64, len = 98 : i64, numPlanes = 32 : i64, srcPlaneStride = 196 : i64, srcStride = 2 : i64, srcWidth = 98 : i64}, port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<32x49xf16, [@DDR, 0]>)
    //CHECK:                outputs([[OUTPUT1]] : memref<49x32xf16, [@CMX_NN, 1]>) -> memref<49x32xf16, [@CMX_NN, 1]>

    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer<1x32x14x7xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x8x8x!qElemType, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @ClusterPermuteDMAWithDistributedInputAndOutput
func.func @ClusterPermuteDMAWithDistributedInputAndOutput() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer "CMX_NN" <2000> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x4x8x8x!qElemType, #NHWC, @CMX_NN>) outputs(%1 as %arg1: memref<1x4x8x8x!qElemType, #NCHW, @CMX_NN>) -> !OutputDistributed {
        VPUIP.PermuteDMA {mem_perm = #map, port = 0 : i64}
              inputs(%arg0 : memref<1x4x8x8x!qElemType, #NHWC, @CMX_NN>)
              outputs(%arg1 : memref<1x4x8x8x!qElemType, #NCHW, @CMX_NN>) -> memref<1x4x8x8x!qElemType, #NCHW, @CMX_NN>
      }
    }
    return %1: !OutputDistributed
    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer "CMX_NN" <2000> -> !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[BUFF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <2000> -> !VPUIP.DistributedBuffer<4x64x!qElemType, affine_map<(d0, d1) -> (d0, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    // CHECK:          VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 1 : i64, dstStride = 64 : i64, dstWidth = 1 : i64, len = 4 : i64, numPlanes = 64 : i64, srcPlaneStride = 4 : i64, srcStride = 1 : i64, srcWidth = 4 : i64}, port = 0 : i64}
    // CHCEK-SAME:           inputs([[INPUT]] : memref<64x4x!qElemType, [@CMX_NN, 0]>) outputs([[BUF]] : !VPUIP.DistributedBuffer<4x64x!qElemType, affine_map<(d0, d1) -> (d0, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<4x64x!qElemType, affine_map<(d0, d1) -> (d0, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    }
    // CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

}
