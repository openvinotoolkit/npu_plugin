// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --unroll-depth-to-space-dma  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @UnrollDepthToSpaceDMABlockFirstNHWC(%input: memref<1x8x2x3xf16, #NHWC>, %output: memref<1x2x4x6xf16, #NHWC>) -> memref<1x2x4x6xf16, #NHWC> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input : memref<1x8x2x3xf16, #NHWC>) outputs(%inBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", output_channel = 2 : i64, output_width = 6 : i64, port = 0 : i64}
                inputs(%inBuffer : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) outputs(%output :  memref<1x2x4x6xf16, #NHWC>) -> memref<1x2x4x6xf16, #NHWC>
    }

    return %output: memref<1x2x4x6xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <152> -> memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64}
    //CHECK:            inputs(%arg0 : memref<1x8x2x3xf16, #NHWC>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", output_channel = 2 : i64, output_width = 6 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_0]] : memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", output_channel = 2 : i64, output_width = 6 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_1]] : memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64}
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x2x4x6xf16, #NHWC>) -> memref<1x2x4x6xf16, #NHWC>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x2x4x6xf16, #NHWC>
}

func @UnrollDepthToSpaceDMADepthFirstNHWC(%input: memref<1x8x2x1xf16, #NHWC>, %output: memref<1x2x4x2xf16, #NHWC>) -> memref<1x2x4x2xf16, #NHWC> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %inBuffer = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    %outBuffer = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.NNDMA {port = 0 : i64} inputs(%input : memref<1x8x2x1xf16, #NHWC>) outputs(%inBuffer : memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", output_channel = 2 : i64, output_width = 2 : i64, port = 0 : i64}
                inputs(%inBuffer : memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%outBuffer : memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%outBuffer : memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>) outputs(%output :  memref<1x2x4x2xf16, #NHWC>) -> memref<1x2x4x2xf16, #NHWC>
    }

    return %output: memref<1x2x4x2xf16, #NHWC>

    //CHECK:    [[BARRIER_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BARRIER_1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <12> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <136> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <130> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUTPUT_3:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <138> -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64}
    //CHECK:            inputs(%arg0 : memref<1x8x2x1xf16, #NHWC>)
    //CHECK:            outputs([[INPUT_BUFFER]] : memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", output_channel = 2 : i64, output_width = 2 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_0]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_0]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", output_channel = 2 : i64, output_width = 2 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_1]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_1]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", output_channel = 2 : i64, output_width = 2 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_2]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_2]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", output_channel = 2 : i64, output_width = 2 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_3]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_3]] : memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x2x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier)  {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64}
    //CHECK:            inputs([[OUTPUT_BUFFER]] : memref<1x2x4x2xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs(%arg1 : memref<1x2x4x2xf16, #NHWC>) -> memref<1x2x4x2xf16, #NHWC>
    //CHECK:    }

    //CHECK:    return %arg1 : memref<1x2x4x2xf16, #NHWC>
}
