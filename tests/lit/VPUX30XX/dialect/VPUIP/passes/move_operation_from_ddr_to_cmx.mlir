// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --move-operation-from-ddr-to-cmx %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveDepthToSpaceFromDDRToCMX_BLOCKS_FIRST
func @MoveDepthToSpaceFromDDRToCMX_BLOCKS_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    %depthToSpace = VPUIP.DepthToSpaceUPA {block_size = 2 : i64, mode = "BLOCKS_FIRST"}
                inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>
    
    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYBUFFER_0:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[COPYOUT_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
    //CHECK:            outputs([[COPYBUFFER_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OPBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", output_channel = 2 : i64, output_width = 6 : i64, port = 0 : i64}
    //CHECK:            inputs([[COPYOUT_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OPBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[COPYBUFFER_1:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYOUT_1:%.*]] = VPUIP.Copy inputs([[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[COPYBUFFER_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    return [[COPYOUT_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveDepthToSpaceFromDDRToCMX_DEPTH_FIRST
func @MoveDepthToSpaceFromDDRToCMX_DEPTH_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    %depthToSpace = VPUIP.DepthToSpaceUPA {block_size = 2 : i64, mode = "DEPTH_FIRST"}
                inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYBUFFER_0:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[COPYOUT_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
    //CHECK:            outputs([[COPYBUFFER_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OPBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "DEPTH_FIRST", output_channel = 2 : i64, output_width = 6 : i64, port = 0 : i64}
    //CHECK:            inputs([[COPYOUT_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OPBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[COPYBUFFER_1:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYOUT_1:%.*]] = VPUIP.Copy inputs([[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[COPYBUFFER_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    return [[COPYOUT_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @MoveMemPermuteFromDDRToCMX(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>)
        -> memref<1x8x16x16xf16, @DDR> {
    %buf1 = memref.alloc() : memref<1x8x16x16xf16, #NCHW, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #map} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%buf1 : memref<1x8x16x16xf16, #NCHW, @DDR>) -> memref<1x8x16x16xf16, @DDR>
    return %0: memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR1]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PermuteDMA {dst_stride = 0 : i64, mem_perm = #map, port = 0 : i64} inputs([[VAR2]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x8x16x16xf16, @DDR>) -> memref<1x8x16x16xf16, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x8x16x16xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func @CanNotMoveMemPermuteFromDDRToCMX(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>)
        -> memref<1x8x16x16xf16, #NHCW, @DDR> {
    %buf = memref.alloc() : memref<1x8x16x16xf16, #NHCW, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #map} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%buf : memref<1x8x16x16xf16, #NHCW, @DDR>) -> memref<1x8x16x16xf16, #NHCW, @DDR>
    return %0: memref<1x8x16x16xf16, #NHCW, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NHCW, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #map}
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x8x16x16xf16, #NHCW, @DDR>) -> memref<1x8x16x16xf16, #NHCW, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x8x16x16xf16, #NHCW, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @CanNotMoveMemPermuteFromDDRToCMXWithLargeDMANumber(%arg0: memref<1x8x256x256xf16, #NHWC, @DDR>)
        -> memref<1x8x256x256xf16, @DDR> {
    %buf = memref.alloc() : memref<1x8x256x256xf16, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #map} inputs(%arg0 : memref<1x8x256x256xf16, #NHWC, @DDR>) outputs(%buf : memref<1x8x256x256xf16, @DDR>) -> memref<1x8x256x256xf16, @DDR>
    return %0: memref<1x8x256x256xf16, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x8x256x256xf16, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #map}
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x256x256xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x8x256x256xf16, @DDR>) -> memref<1x8x256x256xf16, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x8x256x256xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func @CanNotMoveMemPermuteFromDDRToCMXWithLargeLenth(%arg0: memref<1x512x32x32xf16, #NHWC, @DDR>)
        -> memref<1x512x32x32xf16, @DDR> {
    %buf = memref.alloc() : memref<1x512x32x32xf16, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #map} inputs(%arg0 : memref<1x512x32x32xf16, #NHWC, @DDR>) outputs(%buf : memref<1x512x32x32xf16, @DDR>) -> memref<1x512x32x32xf16, @DDR>
    return %0: memref<1x512x32x32xf16, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x512x32x32xf16, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #map}
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x32x32xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x512x32x32xf16, @DDR>) -> memref<1x512x32x32xf16, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x512x32x32xf16, @DDR>
}
