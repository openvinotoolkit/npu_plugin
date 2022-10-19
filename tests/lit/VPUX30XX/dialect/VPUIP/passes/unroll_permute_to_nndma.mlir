// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --unroll-permute-to-nndma  %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNHWCToNCHW
func @PermuteToDMAWithNHWCToNCHW() -> memref<1x8x16x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {dst_stride = 256 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<256x8xf16, [@CMX_NN, 0]>) -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithNCHWToNHWC
func @PermuteToDMAWithNCHWToNHWC() -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64}
                inputs(%input : memref<1x8x16x16xf16, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %output: memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<128x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<128x16xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {dst_stride = 128 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER]] : memref<128x16xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER]] : memref<128x16xf16, [@CMX_NN, 0]>) -> memref<128x16xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteToDMAWithLargePlaneNumber
func @PermuteToDMAWithLargePlaneNumber() -> memref<1x8x32x16xf16, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0: !VPURT.Barrier)  {
        VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, port = 0 : i64}
                inputs(%input : memref<1x8x32x16xf16, #NHWC, [@CMX_NN, 0]>)
                outputs(%output : memref<1x8x32x16xf16, [@CMX_NN, 0]>) -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    }
    return %output: memref<1x8x32x16xf16, [@CMX_NN, 0]>


    //CHECK:    [[BARRIER:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[INPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[RETURN_BUFFER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x8x32x16xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4608> -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_BUFFER0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<256x8xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {dst_stride = 512 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER0]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER0]] : memref<256x8xf16, [@CMX_NN, 0]>) -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    }

    //CHECK:    VPURT.Task updates([[BARRIER]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:        VPUIP.PermuteDMA {dst_stride = 512 : i64, port = 0 : i64}
    //CHECK:            inputs([[INPUT_BUFFER1]] : memref<256x8xf16, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OUTPUT_BUFFER1]] : memref<256x8xf16, [@CMX_NN, 0]>) -> memref<256x8xf16, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    return [[RETURN_BUFFER]] : memref<1x8x32x16xf16, [@CMX_NN, 0]>
}
