//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --unroll-upsampling-dma  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNCHW
func @UnrollUpsamplingDMAWithNCHW() -> memref<1x32x32x64xf16, #NCHW, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x32x16x32xf16, #NCHW, @DDR>
    %output = VPURT.DeclareBuffer "DDR" <262144> -> memref<1x32x32x64xf16, #NCHW, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 65009 : i64, cycleEnd = 87805 : i64, isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x32x16x32xf16, #NCHW, @DDR>)
                        outputs(%output : memref<1x32x32x64xf16, #NCHW, @DDR>) -> memref<1x32x32x64xf16, #NCHW, @DDR>
    }
    return %output: memref<1x32x32x64xf16, #NCHW, @DDR>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x1x256x32xf16, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT1:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <16384> -> memref<1x1x256x32xf16, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer "DDR" <262144> -> memref<1x32x32x64xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "DDR" <262144> -> memref<1x1x512x64xf16, @DDR>
    // CHECK-DAG:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "DDR" <327680> -> memref<1x1x512x64xf16, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) 
    // CHECK-SAME:          isTrailingSWLayer = false
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = {
    // CHECK-SAME:              dstPlaneStride = 256 : i64
    // CHECK-SAME:              dstStride = 4 : i64
    // CHECK-SAME:              dstWidth = 2 : i64
    // CHECK-SAME:              len = 64 : i64
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              srcPlaneStride = 64 : i64
    // CHECK-SAME:              srcStride = 64 : i64
    // CHECK-SAME:              srcWidth = 64 : i64
    // CHECK-SAME:          }
    // CHECK-SAME:          port = 0 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT0]] : memref<1x1x256x32xf16, [@DDR, 0]>) 
    // CHECK-SAME:      outputs([[OUTPUT0]] : memref<1x1x512x64xf16, @DDR>) -> memref<1x1x512x64xf16, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) 
    // CHECK-SAME:          isTrailingSWLayer = false
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = {
    // CHECK-SAME:              dstPlaneStride = 256 : i64
    // CHECK-SAME:              dstStride = 4 : i64
    // CHECK-SAME:              dstWidth = 2 : i64
    // CHECK-SAME:              len = 64 : i64
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              srcPlaneStride = 64 : i64
    // CHECK-SAME:              srcStride = 64 : i64
    // CHECK-SAME:              srcWidth = 64 : i64
    // CHECK-SAME:          }
    // CHECK-SAME:          port = 1 : i64, 
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT1]] : memref<1x1x256x32xf16, [@DDR, 0]>) 
    // CHECK-SAME:      outputs([[OUTPUT1]] : memref<1x1x512x64xf16, @DDR>) -> memref<1x1x512x64xf16, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x32x32x64xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @UnrollUpsamplingDMAWithNHWC
func @UnrollUpsamplingDMAWithNHWC() -> memref<1x16x1024x32xf16, #NHWC, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x512x32xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer "DDR" <524288> -> memref<1x16x1024x32xf16, #NHWC, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 65009 : i64, cycleEnd = 87805 : i64, isTrailingSWLayer = false} {
        %111 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
                        inputs(%input : memref<1x16x512x32xf16, #NHWC, @DDR>)
                        outputs(%output : memref<1x16x1024x32xf16, #NHWC, @DDR>) -> memref<1x16x1024x32xf16, #NHWC, @DDR>
    }

    return %output: memref<1x16x1024x32xf16, #NHWC, @DDR>

    // CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:    [[INPUT0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>
    // CHECK-DAG:    [[INPUT1:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <262144> -> memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>

    // CHECK-DAG:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer "DDR" <524288> -> memref<1x16x1024x32xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer "DDR" <524288> -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK-DAG:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer "DDR" <1572864> -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) 
    // CHECK-SAME:          isTrailingSWLayer = false
    // CHECK-SAME:  {
    // CHECK:           VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = {
    // CHECK-SAME:              dstPlaneStride = 4096 : i64
    // CHECK-SAME:              dstStride = 64 : i64
    // CHECK-SAME:              dstWidth = 32 : i64
    // CHECK-SAME:              len = 1024 : i64
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              srcPlaneStride = 1024 : i64
    // CHECK-SAME:              srcStride = 1024 : i64
    // CHECK-SAME:              srcWidth = 1024 : i64
    // CHECK-SAME:          }
    // CHECK-SAME:          port = 0 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      }
    // CHECK-SAME:      inputs([[INPUT0]] : memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>) 
    // CHECK-SAME:      outputs([[OUTPUT0]] : memref<1x16x512x64xf16, #NHWC, @DDR>) -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK:       }


    // CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) 
    // CHECK-SAME:          isTrailingSWLayer = false
    // CHECK-SAME:  {
    // CHECK:        VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:          dma_descriptor = {
    // CHECK-SAME:              dstPlaneStride = 4096 : i64
    // CHECK-SAME:              dstStride = 64 : i64
    // CHECK-SAME:              dstWidth = 32 : i64
    // CHECK-SAME:              len = 1024 : i64
    // CHECK-SAME:              numPlanes = 256 : i64
    // CHECK-SAME:              srcPlaneStride = 1024 : i64
    // CHECK-SAME:              srcStride = 1024 : i64
    // CHECK-SAME:              srcWidth = 1024 : i64
    // CHECK-SAME:          }
    // CHECK-SAME:          port = 1 : i64
    // CHECK-SAME:          upsampling_factor = [1, 1, 2, 2]
    // CHECK-SAME:      inputs([[INPUT1]] : memref<1x16x256x32xf16, #NHWC, [@DDR, 0]>) 
    // CHECK-SAME:      outputs([[OUTPUT1]] : memref<1x16x512x64xf16, #NHWC, @DDR>) -> memref<1x16x512x64xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:    return [[OUTPUT]] : memref<1x16x1024x32xf16, #NHWC, @DDR>
}
