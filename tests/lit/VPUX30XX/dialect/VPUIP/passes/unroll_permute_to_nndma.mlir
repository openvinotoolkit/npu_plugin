//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --unroll-permute-to-nndma  %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNHWCToNCHW
func.func @PermuteToDMAWithNHWCToNCHW() -> memref<1x8x16x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 512 : i64, dstPlaneStride = 2 : i64>
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

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<128x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<16x128xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:        dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 128 : i64, len = 32 : i64, srcWidth = 32 : i64, srcStride = 2 : i64, srcPlaneStride = 32 : i64, dstWidth = 2 : i64, dstStride = 256 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:        port = 0 : i64}
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

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x1x32xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x32x1x8xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x32xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 32 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 64 : i64, dstPlaneStride = 2 : i64>
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

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64, src_plane_stride = 0 : i64}
                inputs(%input : memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x32x16xf16, [@CMX_NN, 0]>) -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x8x32x16xf16, [@CMX_NN, 0]>


    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4608> -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<8x256xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 1024 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAME:       port = 0 : i64}
    //CHECK:       inputs([[INPUT_BUFFER0]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER0]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {
    //CHECK-SAME:       dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 256 : i64, len = 16 : i64, srcWidth = 16 : i64, srcStride = 2 : i64, srcPlaneStride = 16 : i64, dstWidth = 2 : i64, dstStride = 1024 : i64, dstPlaneStride = 2 : i64>
    //CHECK-SAMEL       port = 0 : i64}
    //CHECK-SAME:       inputs([[INPUT_BUFFER1]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUTPUT_BUFFER1]] : memref<8x256xf16, [@CMX_NN, 0]>) -> memref<8x256xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x32x16xf16, [@CMX_NN, 0]>
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
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <2000> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
        VPUIP.PermuteDMA {mem_perm = #map, port = 0 : i64}
            inputs(%0 : !InputDistributed)
            outputs(%1 : !OutputDistributed) -> !OutputDistributed
    }
    return %1: !OutputDistributed
    // CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x4x!qElemType, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <2000> -> !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[BUFF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000> -> !VPUIP.DistributedBuffer<4x64x!qElemType, affine_map<(d0, d1) -> (d0, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    // CHECK:          VPUIP.PermuteDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 64 : i64, len = 4 : i64, srcWidth = 4 : i64, srcStride = 1 : i64, srcPlaneStride = 4 : i64, dstWidth = 1 : i64, dstStride = 64 : i64, dstPlaneStride = 1 : i64>, port = 0 : i64}
    // CHCEK-SAME:           inputs([[INPUT]] : memref<64x4x!qElemType, [@CMX_NN, 0]>) outputs([[BUF]] : !VPUIP.DistributedBuffer<4x64x!qElemType, affine_map<(d0, d1) -> (d0, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<4x64x!qElemType, affine_map<(d0, d1) -> (d0, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    }
    // CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x8x8x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

}
