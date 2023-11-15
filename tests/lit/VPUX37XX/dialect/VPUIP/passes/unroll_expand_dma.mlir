//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --unroll-expand-dma  %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x320xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollExpandDMAWithSEGMENTED
func.func @UnrollExpandDMAWithSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
                inputs(%input : memref<1x3x240x320xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <230400> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4432x1x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollExpandDMAWithDUPLICATED
func.func @UnrollExpandDMAWithDUPLICATED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4420x1x2xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
                inputs(%input : memref<1x4420x1x2xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4420x1x2xf16, #NHWC, @DDR>
    //CHECK:    [[RETURN:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 17680 : i64, srcWidth = 17680 : i64, srcStride = 17680 : i64, srcPlaneStride = 0 : i64, dstWidth = 8840 : i64, dstStride = 8864 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT]] : memref<1x4420x1x2xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT]] : !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:                    -> !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    return [[RETURN]] : !VPUIP.DistributedBuffer<1x4432x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>  
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x4x240x320xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [2, 2],
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollExpandDMAWithOVERLAPPED
func.func @UnrollExpandDMAWithOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x240x320xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
                inputs(%input : memref<1x3x240x320xf16, #NHWC, @DDR>)
                outputs(%output : !OutputDistributed) -> !OutputDistributed
    }

    return %output: !OutputDistributed

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x121x320xf16, #NHWC, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <230400> -> memref<1x3x120x320xf16, #NHWC, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>

    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x121x320xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 232320 : i64, srcWidth = 232320 : i64, srcStride = 232320 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x3x121x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x4x121x320xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x121x320xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 230400 : i64, srcWidth = 230400 : i64, srcStride = 230400 : i64, srcPlaneStride = 0 : i64, dstWidth = 6 : i64, dstStride = 8 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x3x120x320xf16, #NHWC, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x4x120x320xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    return [[OUTPUT]] : !VPUIP.DistributedBuffer<1x4x240x320xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 0.0040670955882352944:128>

// CHECK-LABEL: @UnrollExpandDMAWithLargeSizeAndDiffWithExpandAxis
func.func @UnrollExpandDMAWithLargeSizeAndDiffWithExpandAxis() -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <0> -> memref<1x20x720x1280x!qElemType, #NWCH, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
                inputs(%input : memref<1x20x720x1280x!qElemType, #NWCH, @DDR>)
                outputs(%output : memref<1x32x720x1280x!qElemType, #NWCH, @DDR>)
                -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>
    }

    return %output: memref<1x32x720x1280x!qElemType, #NWCH, @DDR>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x20x720x1165x!qElemType, #NWCH, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <16776000> -> memref<1x20x720x115x!qElemType, #NWCH, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <45273600> -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 16776000 : i64, srcWidth = 16776000 : i64, srcStride = 16776000 : i64, srcPlaneStride = 0 : i64, dstWidth = 14400 : i64, dstStride = 23040 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x20x720x1165x!qElemType, #NWCH, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x32x720x1280x!qElemType, #NWCH, @DDR>) -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 1656000 : i64, srcWidth = 1656000 : i64, srcStride = 1656000 : i64, srcPlaneStride = 0 : i64, dstWidth = 14400 : i64, dstStride = 23040 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x20x720x115x!qElemType, #NWCH, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x32x720x1280x!qElemType, #NWCH, @DDR>) -> memref<1x32x720x1280x!qElemType, #NWCH, @DDR>

    //CHECK:    return [[OUTPUT]] : memref<1x32x720x1280x!qElemType, #NWCH, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0040670955882352944:128>

// CHECK-LABEL: @UnrollExpandDMAWithLargeSizeAndSameWithExpandAxis
func.func @UnrollExpandDMAWithLargeSizeAndSameWithExpandAxis() -> memref<1x32x720x1280x!qElemType, #NCHW, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <DDR> <0> -> memref<1x20x720x1280x!qElemType, #NCHW, @DDR>
    %output = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x32x720x1280x!qElemType, #NCHW, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
                inputs(%input : memref<1x20x720x1280x!qElemType, #NCHW, @DDR>)
                outputs(%output : memref<1x32x720x1280x!qElemType, #NCHW, @DDR>)
                -> memref<1x32x720x1280x!qElemType, #NCHW, @DDR>
    }

    return %output: memref<1x32x720x1280x!qElemType, #NCHW, @DDR>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x18x720x1280x!qElemType, @DDR>
    //CHECK:    [[INPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <16588800> -> memref<1x2x720x1280x!qElemType, @DDR>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x32x720x1280x!qElemType, @DDR>
    //CHECK:    [[OUTPUT0:%.*]] = VPURT.DeclareBuffer <DDR> <18432000> -> memref<1x32x720x1280x!qElemType, @DDR>
    //CHECK:    [[OUTPUT1:%.*]] = VPURT.DeclareBuffer <DDR> <35020800> -> memref<1x32x720x1280x!qElemType, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 16588800 : i64, srcWidth = 16588800 : i64, srcStride = 16588800 : i64, srcPlaneStride = 0 : i64, dstWidth = 16588800 : i64, dstStride = 29491200 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 0 : i64}
    //CHECK:                inputs([[INPUT0]] : memref<1x18x720x1280x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT0]] : memref<1x32x720x1280x!qElemType, @DDR>) -> memref<1x32x720x1280x!qElemType, @DDR>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.ExpandDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 1 : i64, len = 1843200 : i64, srcWidth = 1843200 : i64, srcStride = 1843200 : i64, srcPlaneStride = 0 : i64, dstWidth = 1843200 : i64, dstStride = 29491200 : i64, dstPlaneStride = 0 : i64>, pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0], port = 1 : i64}
    //CHECK:                inputs([[INPUT1]] : memref<1x2x720x1280x!qElemType, @DDR>)
    //CHECK:                outputs([[OUTPUT1]] : memref<1x32x720x1280x!qElemType, @DDR>) -> memref<1x32x720x1280x!qElemType, @DDR>

    //CHECK:    return [[OUTPUT]] : memref<1x32x720x1280x!qElemType, @DDR>
}
